# coding: utf-8
import os
from collections import OrderedDict
import logging
import argparse
import random


import netCDF4 as nc4
import pandas as pd
import numpy as np
import pvlib as pv
import scipy.io


from satoi import oi


DROP_SENSORS = []#[410, 424, 432, 446]


def suny_ghi(lats, lons, raw, terrain_elevation):
    pressure = pv.atmosphere.alt2pres(terrain_elevation)
    app_zenith = None
    zenith = None
    elevation = None
    azimuth = None
    for atime in raw.index:
        res = pv.spa.solar_position(
            atime.value/10**9, lats, lons, terrain_elevation, pressure,
            12, 67.0, 0.)
        if app_zenith is None:
            app_zenith = pd.DataFrame({atime: res[0]})
        else:
            app_zenith[atime] = res[0]
        if zenith is None:
            zenith = pd.DataFrame({atime: res[1]})
        else:
            zenith[atime] = res[1]
        if elevation is None:
            elevation = pd.DataFrame({atime: res[3]})
        else:
            elevation[atime] = res[3]
        if azimuth is None:
            azimuth = pd.DataFrame({atime: res[4]})
        else:
            azimuth[atime] = res[4]
    cos_zen = np.cos(np.radians(zenith.T))
    am = pv.atmosphere.absoluteairmass(pv.atmosphere.relativeairmass(
        app_zenith.T), pressure.values)
    soldist = pv.solarposition.pyephem_earthsun_distance(raw.index).T.pow(2)

    raw = np.sqrt(raw * cos_zen) * 255
    norpix = (raw * am).mul(soldist, axis=0)
    upper_bound = []
    lower_bound = []
    for c in norpix.columns:
        upper_bound.append(norpix[c].nlargest(20).mean())
        lower_bound.append(norpix[c].nsmallest(20).mean())
    low = pd.Series(lower_bound)
    up = pd.Series(upper_bound)

    CI = (norpix - low) / (up - low)
    KTM = (2.36 * CI.pow(5) - 6.2 * CI.pow(4) + 6.22 * CI.pow(3) -
           2.63 * CI.pow(2) - 0.58 * CI + 1)
    ones = pd.DataFrame(np.ones(raw.shape), index=raw.index,
                        columns=raw.columns)
    pvlib_path = os.path.dirname(os.path.abspath(pv.__file__))
    filepath = os.path.join(pvlib_path, 'data', 'LinkeTurbidities.mat')
    mat = scipy.io.loadmat(filepath)
    table = mat['LinkeTurbidity']
    lat_inds = np.around(pv.clearsky._linearly_scale(lats, 90, -90, 1,
                                                     2160).values).astype(int)
    lon_inds = np.around(pv.clearsky._linearly_scale(lons, -180, 180, 1,
                                                     4320).values).astype(int)

    g = table[lat_inds, lon_inds].T
    linke_turbidity = ones.mul(pd.Series(raw.index.month, index=raw.index),
                               axis=0)
    linke_turbidity = linke_turbidity.apply(lambda x: g[x[0] - 1], axis=1)
    linke_turbidity /= 20.
    TL = linke_turbidity

    telev = ones.mul(terrain_elevation, axis=1)
    cg1 = (0.0000509 * telev + 0.868)
    cg2 = (0.0000392 * telev + 0.0387)
    I0 = ones.mul(pv.irradiance.extraradiation(raw.index), axis=0)
    fh1 = np.exp(-1.0 / 8000 * telev)
    fh2 = np.exp(-1.0 / 1250 * telev)
    Ghcnew = cg1 * I0 * cos_zen * np.exp(-1.0 * cg2 * am * (
        fh1 + fh2 * (TL - 1))) * np.exp(0.01 * am**1.8)

    GHI = KTM * Ghcnew * (0.0001 * KTM * Ghcnew + 0.9)
    return GHI, Ghcnew


def main(input_path, satellite_path, outfile, label, suny, times_file=None):
    # ## Get the satellite data
    sat_dataset = nc4.Dataset(satellite_path, mode='r')
    times = nc4.chartostring(sat_dataset.variables['Times'][:]).astype('U')
    sat_times = pd.to_datetime(times, utc=True, format='%Y-%m-%d_%H:%M:%S')

    # ## get sensor metadata
    logging.info('Loading metadata')
    raw_metadata = pd.read_csv(os.path.join(input_path, 'metadata.data'),
                               sep='\t',
                               header=None)
    metadata = raw_metadata.iloc[:, [0, 3, 8, 9, 11]]
    metadata.columns = ['ID', 'Sensor Type', 'Latitude', 'Longitude',
                        'Measurement Units']
    metadata = metadata.set_index('ID')

    # ## get the clearsky-profile data
    logging.info('Getting clearsky-profiles')
    # rawclrs = pd.read_csv(os.path.join(input_path, 'clearsky.data'),
    #                       sep='\t',
    #                       header=None)
    # rawclrs.columns = ['Id', 'Timestamp', 'Clearsky-Profile']
    # times = pd.to_datetime(rawclrs['Timestamp'], utc=True, unit='s')
    # rawclrs['Timestamp'] = times
    # rawclrs = rawclrs.pivot_table(values='Clearsky-Profile', index='Timestamp',
    #                               columns='Id')

    # clrs = OrderedDict()
    # for name, series in rawclrs.iteritems():
    #     series = series.dropna().tz_localize('UTC')
    #     interpolator = interp1d(series.index.values.astype(int), series.values,
    #                             bounds_error=False)
    #     resampled = pd.Series(interpolator(sat_times.values.astype(float)),
    #                           index=sat_times)
    #     clrs[name] = resampled
    # clrs = pd.DataFrame(clrs)
    clrs = pd.read_hdf(os.path.join(input_path, 'clearsky.h5'),
                       '/clearsky_profiles')

    # drop the times where nothing has a clear-sky and when sensors
    # don't have all data
    clrs = clrs.dropna().dropna(axis=1, how='any')

    # ## get the measurement data
    logging.info('Getting measurement data')
    rawmeas = pd.read_csv(os.path.join(input_path, 'measurement.data'),
                          sep='\t',
                          header=None)
    rawmeas.columns = ['Id', 'Timestamp', 'Measurement']
    times = pd.to_datetime(rawmeas['Timestamp'], utc=True, unit='s')
    rawmeas['Timestamp'] = times
    rawmeas = rawmeas.pivot_table(values='Measurement', index='Timestamp',
                                  columns='Id')
    meas = OrderedDict()
    index = clrs.index
    for name, series in rawmeas.ix[:, clrs.columns].iteritems():
        series = series.dropna().tz_localize('UTC')
        #resampled = series.resample('1min').mean().resample('5min').first()
        resampled = series.resample('5min', label='last').mean()
        #resampled = pd.rolling_mean(series.resample('1min').mean().interpolate(),
        #                            10, 2, center=True)
        index = resampled.index.intersection(index)
        meas[name] = resampled.ix[index]

    meas2 = OrderedDict()
    for name, series in meas.items():
        meas2[name] = meas[name].ix[index]

    # may be able to get rid of the last dropna and account for nans in the
    # OI routine
    meas = pd.DataFrame(meas2).dropna(axis=1, how='all').dropna(axis=0,
                                                                how='any')

    common_columns = meas.columns.intersection(clrs.columns).drop(DROP_SENSORS)

    # drop times when the clearsky is very high for some reason
    times_to_drop = clrs[clrs > 1e5].dropna(axis=0, how='all').index

    # filter by solar zenith angle
    solpos = pv.solarposition.get_solarposition(clrs.index, 32.2, -110.95, 700)
    times_to_drop = times_to_drop.union(solpos.index[
        solpos['apparent_zenith'] > 60])

    # drop when clrs too small
    times_to_drop = times_to_drop.union(
        clrs[clrs < 1].ix[:, common_columns].dropna(
            axis=0, how='all').index)

    common_times = clrs.index.drop(times_to_drop).intersection(
        sat_times).intersection(meas.index)

    common_columns = common_columns[
        ~((meas / clrs).ix[common_times, common_columns] > 3).any()]

    time_ser = pd.DataFrame(sat_times.to_series())
    time_ser['ind'] = np.arange(0, len(sat_times))
    nc_positions = time_ser['ind'].ix[common_times].values

    # ## store the data into one HDF5 file
    logging.info('Storing data')
    new_store = pd.HDFStore(outfile, complib='zlib',
                            complevel=1, fletcher32=True, mode='w')

    long_ser = pd.Series(sat_dataset.variables['XLONG'][:].ravel())
    lat_ser = pd.Series(sat_dataset.variables['XLAT'][:].ravel())

    meas = meas.ix[common_times, common_columns]
    clrs = clrs.ix[common_times, common_columns]
    meas.columns = [int(a) for a in meas.columns]
    clrs.columns = [int(a) for a in clrs.columns]

    metadata = metadata.ix[common_columns]

    new_store.put('/measurements', meas )
    new_store.put('/clearsky_profiles', clrs)
    new_store.put('/sensor_metadata', metadata)

    new_store.put('/satellite/XLONG', long_ser)
    new_store.put('/satellite/XLAT', lat_ser)

    for var in sat_dataset.variables:
        if var in ('Times', 'XLONG', 'XLAT'):
            continue
        data = sat_dataset.variables[var][nc_positions].astype('float64')
        df = pd.DataFrame(data.reshape(data.shape[0],
                                       data.shape[1] * data.shape[2]),
                          index=common_times)
        new_store.put('/satellite/{}'.format(var), df)

    # Use uasibs to get clear/cloudy times
    sat_ghi = new_store.get('/satellite/GHI').astype('float64')
    sat_ghi_clr = new_store.get('/satellite/CLEAR_GHI')

    sat_ci = sat_ghi.truediv(sat_ghi_clr)
    if times_file is None:
        clear_times = oi.clear_times(sat_ci, meas.truediv(clrs))
        cloudy_times = (~clear_times).nonzero()[0]
        clear_times = clear_times.nonzero()[0]

        clr_len = len(clear_times)
        cloud_len = len(cloudy_times)
        clear_samples = clear_times[random.sample(range(clr_len), clr_len // 3)]
        cloudy_samples = cloudy_times[random.sample(range(cloud_len),
                                                    cloud_len // 3)]
        random_clear_times = sat_ci.index[clear_samples]
        random_times = np.concatenate([clear_samples, cloudy_samples]).flatten()
        new_store.put('/randomly_sampled_times',
                      pd.Series(random_times,
                                index=sat_ci.index[random_times]))
        new_store.put('/randomly_clear_times',
                      pd.Series(clear_samples,
                                index=sat_ci.index[clear_samples]))
        new_store.put('/randomly_cloudy_times',
                      pd.Series(cloudy_samples,
                                index=sat_ci.index[cloudy_samples]))
        new_store.put('/clear_times', sat_ci.index[clear_times].to_series())
        new_store.put('/cloudy_times', sat_ci.index[cloudy_times].to_series())
    else:
        time_store = pd.HDFStore(times_file, 'r')
        for arg in ('/randomly_sampled_times', '/randomly_clear_times',
                    '/randomly_cloudy_times', 'clear_times', 'cloudy_times'):
            new_store.put(arg, time_store.get(arg))
        random_clear_times = time_store.get('/randomly_clear_times').index
        time_store.close()

    if suny:
        pa = new_store.get('/satellite/PA')
        elevation = pd.read_hdf(
            os.path.join(input_path, 'elevation.h5'), 'elevation')
        sat_ghi, sat_ghi_clr = suny_ghi(lat_ser, long_ser, pa, elevation)
        new_store.put('/satellite/GHI', sat_ghi)
        new_store.put('/satellite/CLEAR_GHI', sat_ghi_clr)
        sat_ci = sat_ghi.truediv(sat_ghi_clr)

    # calculate R and D from clear days only
    full_ci = meas.truediv(clrs).ix[random_clear_times]
    R = np.diag((full_ci - 1).cov().copy())
    R.flags.writeable = True
    R[R < 0.001] = 0.001
    R = np.diag(R)
    R = pd.DataFrame(R, columns=full_ci.columns, index=full_ci.columns)
    new_store.put('/R', R)

    scstd = (sat_ci - 1).ix[random_clear_times].std()
    sv = scstd[scstd > 0].mean()
    D = pd.DataFrame(np.diag(np.ones(scstd.shape) * sv), index=sat_ci.columns,
                     columns=sat_ci.columns)
    new_store.put('/D', D)

    for var in sat_dataset.variables:
        if var in ('Times', 'XLONG', 'XLAT', 'GHI', 'CLEAR_GHI'):
            continue
        df = new_store.get('/satellite/{}'.format(var))
        new_store.put('/satellite/clear_mean_{}'.format(var),
                      df.ix[random_clear_times].mean())
    # add the label
    new_store._handle.root._v_attrs.sumatra_label = label
    new_store.close()


def get_parser():
    argparser = argparse.ArgumentParser(
        description='Convert the raw data to a format for satOI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-v', '--verbose', action='count')
    argparser.add_argument('--label', help='optional label to add to file')
    argparser.add_argument('--suny', help='Calculat GHI using modified SUNY',
                           action='store_true')
    argparser.add_argument(
        '--times-file',
        help='HDF5 file that already has clear/cloudy and random times assigned')
    argparser.add_argument('input_path', help='Directory with raw data csvs')
    argparser.add_argument('satellite_file',
                           help='H5 file with satellite data')
    argparser.add_argument('outfile',
                           help='Path to write combined output data')
    return argparser


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')
    argparser = get_parser()
    args = argparser.parse_args()
    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)
    main(args.input_path, args.satellite_file, args.outfile, args.label,
         args.suny, args.times_file)
