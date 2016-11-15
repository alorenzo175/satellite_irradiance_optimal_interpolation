import os
import sys
import logging
import argparse
import time
import atexit
import hashlib


import pandas as pd
import numpy as np


import satoi
from satoi import oi
VERSION = satoi.__version__


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception",
                  exc_info=(exc_type, exc_value, exc_traceback))


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Run the optimal interpolation on some data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-v', '--verbose', action='count')
    argparser.add_argument('-g', '--gpu', help='GPU to use',
                           type=int, choices=(0, 1))
    argparser.add_argument('-p', '--path', help='Path to the data',
                           default='/storage/ssd/goes_oi/sat_and_obs.h5')
    argparser.add_argument('-s', '--save-dir', help='Path to save the file to',
                           default=os.path.expanduser('~/oi_data/'))
    argparser.add_argument('--old-correlation', action='store_true')
    argparser.add_argument('--correlation-variable', default='PA',
                           help='Variable to use to make correlation between pixels')  # NOQA
    argparser.add_argument('--keep-sensors', help='DG sensors to keep')
    argparser.add_argument('--witheld-sensors',
                           help='Sensors to withold and use to calculate MSE/tr(M)')  # NOQA
    argparser.add_argument('--d-inflation', help='Multiplier to apply to D',
                           type=float, default=2.0)
    argparser.add_argument('--cloud-radius',
                           help='Distance a cloud edge must be from a sensor to use it')  # NOQA
    argparser.add_argument('--shift-clouds', action='store_true',
                           help='Determine a shift to minimize MSE',)
    argparser.add_argument('--no-shift', action='store_true',
                           help='Dont shift clouds')
    argparser.add_argument('--clear-d-scale',
                           help='Scale inflation by this for clear days')
    argparser.add_arguement('--dtype')
    argparser.add_argument('--empirical-P', action='store_true')
    argparser.add_argument('--spatial-correlation', action='store_true')
    argparser.add_argument('--analyze-sensors',
                           help='Sensors to make plots for',
                           nargs='+', default=[])
    argparser.add_argument(
        '--random-times',
        help='If a random subset vs all times should be used',
        default='sampled')
    argparser.add_argument('--nearby-points',
                           help='Number of points to use when constructing H',
                           default=1)
    argparser.add_argument(
        'length', help='Characterstic length for exponential correlation',
        type=float)
    argparser.add_argument(
        'gamma', type=float,
        help='Exponent in exponential (zero means use linear correlation)')
    args = argparser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)
    return args


class CalcOI(oi.OI):
    _spatial_C = None

    def __init__(self, args):
        if args.gpu is not None:
            self.using_gpu = True
            import pycuda.driver
            import pycuda.gpuarray as gpuarray
            self.gpuarray = gpuarray
            import skcuda.linalg as culinalg
            self.culinalg = culinalg
            pycuda.driver.init()
            dev = pycuda.driver.Device(args.gpu)
            self.context = dev.make_context()
            atexit.register(self.finish_up)
            logging.info('Running on GPU %s', args.gpu)
            super().__init__('gpu', self.context)
        else:
            self.using_gpu = False
            super().__init__('cpu')

        for key, val in vars(args).items():
            setattr(self, key, val)

        self.k = self.nearby_points
        self.withheld = self.witheld_sensors or []
        self.keep = self.keep_sensors or [414, 427, 449,
                                          433, 404, 428, 450, 416]
        self.dtype = args.dtype or 'float32'

    def transpose(self, *args, **kwargs):
        if self.using_gpu:
            return self.culinalg.transpose(*args, **kwargs).astype(self.dtype).copy()
        else:
            return np.transpose(*args, **kwargs).copy()

    def dot(self, *args, **kwargs):
        if self.using_gpu:
            return self.culinalg.dot(*args, **kwargs).copy()
        else:
            return np.dot(*args, **kwargs).copy()

    def trace(self, *args, **kwargs):
        if self.using_gpu:
            return self.culinalg.trace(*args, **kwargs).copy()
        else:
            return np.trace(*args, **kwargs).copy()

    def diag(self, *args, **kwargs):
        if self.using_gpu:
            return self.culinalg.diag(*args, **kwargs).copy()
        else:
            return np.diag(*args, **kwargs).copy()

    def to_gpu(self, a):
        if self.using_gpu:
            return self.gpuarray.to_gpu(a.copy().astype(self.dtype))
        else:
            return a

    def from_gpu(self, a):
        if self.using_gpu:
            return a.get()
        else:
            return a

    def finish_up(self):
        self.context.pop()
        self.context = None
        from pycuda.tools import clear_context_caches
        clear_context_caches()

    def compute_Hs(self):
        self.Hp = self.to_gpu(self.compute_H(
            self.sensor_lats, self.sensor_lons,
            self.sat_lats, self.sat_lons, self.k))

        self.H = self.to_gpu(
            self.compute_H(self.dropped_sensor_lats,
                           self.dropped_sensor_lons,
                           self.sat_lats, self.sat_lons, self.k))
        self.H_wh = self.to_gpu(
            self.compute_H(self.sensor_lats[self.withheld],
                           self.sensor_lons[self.withheld],
                           self.sat_lats, self.sat_lons, self.k))
        self.HT = self.transpose(self.H)
        self.HT_wh = self.transpose(self.H_wh)
        self.Hzeros = self.to_gpu(np.zeros(self.H.shape[1]))

        if self.cloud_radius is not None:
            self.nearby_inds = self.calc_nearby_points(
                self.dropped_sensor_lats,
                self.dropped_sensor_lons,
                self.sat_lats, self.sat_lons,
                self.cloud_radius)
        else:
            self.nearby_inds = None

    def load_data(self):
        with open(self.path, 'rb') as f:
            self.dataset_md5 = hashlib.md5(f.read()).hexdigest()
        dataset = pd.HDFStore(self.path, mode='r')
        sensor_metadata = dataset['sensor_metadata']
        self.sensor_lats = sensor_metadata['Latitude']
        self.sensor_lons = sensor_metadata['Longitude']
        withheld = self.withheld or []
        self.drop_sens = withheld.copy()
        keep = self.keep or [414, 427, 449, 433, 404, 428, 450, 416]
        self.drop_sens.extend([
            a for a in sensor_metadata.index if a >= 400 and a < 500 and
            a not in keep])
        self.dropped_sensor_lats = self.sensor_lats.drop(self.drop_sens,
                                                         inplace=False).copy()
        self.dropped_sensor_lons = self.sensor_lons.drop(self.drop_sens,
                                                         inplace=False).copy()

        self.sat_lats = dataset.get('/satellite/XLAT')
        self.sat_lons = dataset.get('/satellite/XLONG')

        sat_ci = dataset.get('/satellite/GHI').truediv(
            dataset.get('/satellite/CLEAR_GHI'))

        full_ci = dataset['measurements'].truediv(dataset['clearsky_profiles'])
        ci = full_ci.drop(self.drop_sens, axis=1)

        self.clear_times = dataset.get('/clear_times').index
        self.cloudy_times = dataset.get('/cloudy_times').index

        sat_cpa = dataset.get('satellite/{}'.format(self.correlation_variable))
        cpa = sat_cpa - dataset.get('/satellite/clear_mean_{}'.format(
            self.correlation_variable))

        if self.random_times is not None:
            logging.info('Using randomly {} times'.format(self.random_times))
            random_times = dataset.get('/randomly_{}_times'.format(
                self.random_times))
            self.times = random_times.index
        else:
            self.times = sat_ci.index

        random_sampled_times = dataset.get('/randomly_sampled_times')

        self.sensor_data = full_ci.ix[self.times, self.analyze_sensors]
        full_R = dataset['R']
        R_wh = full_R.ix[withheld, withheld].values
        R = full_R.drop(self.drop_sens).drop(
            self.drop_sens, axis=1).values
        # D here actually refers to the std deviation matrix or D^1/2
        D = self.d_inflation * dataset['D'].values
        # assert that D is diagonal to save time computing P
        #if not np.allclose(np.diag(np.diag(D)), D):
        #    raise TypeError('D must be a diagonal matrix')

        self.D = self.to_gpu(D)
        dataset.close()

        xb = sat_ci.ix[self.times]
        self.xbb = xb.copy()
        self.xb = self.to_gpu(xb.values)
        self.cpa = self.to_gpu(cpa.ix[self.times].values)
        if self.empirical_P:
            logging.info('Using empirical P')
            self.P = self.to_gpu(sat_ci.ix[random_sampled_times].cov().values)
        self.y = self.to_gpu(ci.ix[self.times].values)
        self.trR = np.trace(R)
        self.trR_wh = np.trace(R_wh)
        self.R = self.to_gpu(R)

    def compute_C(self, cpa):
        if self.spatial_correlation:
            if self._spatial_C is None:
                self._spatial_C = self.to_gpu(self.distance_correlation(
                    self.sat_lats, self.sat_lons, self.gamma, self.length))
            return self._spatial_C

        if self.gamma == 0:
            return self.linear_corr(cpa, self.length)
        else:
            return self.gamma_exponential_correlation(cpa, self.length,
                                                      self.gamma)

    def make_outputs(self, ldim):
        self.out_Hc = np.empty((ldim, int(len(self.times))))
        self.out_xhat = self.to_gpu(np.empty((ldim, self.xb.shape[0],
                                              self.xb.shape[1])))
        self.out_xhat_at_sens = self.to_gpu(np.empty((ldim,
                                                      int(len(self.times)),
                                                      self.Hp.shape[0])))
        self.out_back_at_sens = self.to_gpu(np.empty((ldim,
                                                      int(len(self.times)),
                                                      self.Hp.shape[0])))
        self.out_phat = self.to_gpu(np.empty((ldim, self.xb.shape[0],
                                              self.xb.shape[1])))
        self.out_trHPH = np.empty((ldim, self.xb.shape[0], 6))

    def compute_and_store(self, i, xb_i, y_i, R, P, H, HT, Hp, Hc, H_wh, HT_wh,
                          li):
        """Compute the analysis and store the result"""
        HPH = self.compute_HPH(P, H, HT)
        xhat, Phat = self.compute_analysis(xb_i, y_i, R, P, H, HT, HPH)

        HPH_wh = self.compute_HPH(P, H_wh, HT_wh)
        HPaH = self.compute_HPH(Phat, H, HT)
        HPaH_wh = self.compute_HPH(Phat, H_wh, HT_wh)

        trHPH = self.trace(HPH)
        trHPH_wh = self.trace(HPH_wh)
        trHPaH = self.trace(HPaH)
        trHPaH_wh = self.trace(HPaH_wh)

        Hpxhat = self.dot(Hp, xhat)[:, 0]
        Hpxb = self.dot(Hp, xb_i)[:, 0]
        diagP = self.diag(Phat)
        traces = np.array((trHPH, trHPH_wh, trHPaH, trHPaH_wh,
                           self.trR, self.trR_wh))

        def put(l):
            self.out_Hc[l, i] = Hc
            self.out_xhat[l, i] = xhat[:, 0]
            self.out_phat[l, i] = diagP
            self.out_xhat_at_sens[l, i] = Hpxhat
            self.out_back_at_sens[l, i] = Hpxb
            self.out_trHPH[l, i] = traces

        if isinstance(li, tuple):
            for l in li:
                put(l)
        else:
            put(li)

    def loop_func(self, i):
        """
        Runs through and computes the analysis after corrections for
        each image
        """
        thetime = self.times[i]
        logging.debug('Running for %s', thetime)

        cpa_i = self.cpa[i, :, None].copy()
        xb_i = self.xb[i, :, None]
        y_i = self.y[i, :, None]
        R = self.R
        if not self.empirical_P:
            if thetime in self.clear_times:
                D = self.D.copy() * self.clear_d_scale
            else:
                D = self.D.copy()

            C = self.compute_C(cpa_i).copy()
            P = self.compute_P(C, D)
        else:
            P = self.P.copy()

        # only do the computation once since no shifting involved at all
        if thetime in self.clear_times:
            self.compute_and_store(i, xb_i.copy(), y_i.copy(), R.copy(),
                                   P.copy(),
                                   self.H.copy(), self.HT.copy(),
                                   self.Hp.copy(), 0, self.H_wh.copy(),
                                   self.HT_wh.copy(), (0, 1))
            return

        if self.shift_clouds:
            H, HT, Hp, Hc = self.minimize_analysis_mse(
                thetime, self.sat_lats, self.sat_lons, self.sensor_lats,
                self.sensor_lons, self.dropped_sensor_lats,
                self.dropped_sensor_lons, self.k, self.cloud_radius,
                xb_i.copy(), y_i.copy(), R.copy(), P.copy(), self.dtype)
            H_wh, HT_wh, _ = self.analysis_mse_minimizer(
                Hc, thetime, self.sat_lats, self.sat_lons, self.sensor_lats,
                self.sensor_lons, self.dropped_sensor_lats,
                self.dropped_sensor_lons, self.cloud_radius, self.k,
                xb_i.copy(), y_i.copy(), R.copy(), P.copy(),
                dtype=self.dtype, retH=True)
            logging.debug('Shiting using a height of %s', Hc)
            self.compute_and_store(i, xb_i.copy(), y_i.copy(), R.copy(),
                                   P.copy(), H.copy(), HT.copy(), Hp.copy(),
                                   Hc, H_wh.copy(),
                                   HT_wh.copy(), 0)

        if self.no_shift:
            if self.cloud_radius is not None:
                H = self.H.copy()
                for j in range(self.nearby_inds.shape[0]):
                    nby_pts = self.xbb.ix[thetime].values[
                        self.nearby_inds[j]].copy()
                    if nby_pts.shape[0] == 0:
                        max_diff = 1
                    else:
                        max_diff = nby_pts.max() - nby_pts.min()
                    if max_diff >= 0.2:
                        H[j] = self.Hzeros.copy()
                HT = self.transpose(H)
            else:
                H = self.H.copy()
                HT = self.HT.copy()

            self.compute_and_store(i, xb_i.copy(), y_i.copy(), R.copy(),
                                   P.copy(), H.copy(), HT.copy(),
                                   self.Hp.copy(), 0,
                                   self.H_wh.copy(), self.HT_wh.copy(),
                                   1)

    def run(self):
        """Run through the loop and save the output"""
        start = time.time()
        for i in range(int(len(self.times))):
            self.loop_func(i)
        end = time.time()
        logging.info('OI took {:0.2f} seconds'.format(end - start))
        xhat_at_sens = self.from_gpu(self.out_xhat_at_sens)
        back_at_sens = self.from_gpu(self.out_back_at_sens)
        out_xhat = self.from_gpu(self.out_xhat)
        out_phat = self.from_gpu(self.out_phat)
        out_trHPH = self.out_trHPH
        out_Hc = self.out_Hc

        output_path = os.path.join(self.save_dir, 'oi_output.h5')
        logging.info('Output path is %s', output_path)
        if not os.path.isdir(os.path.dirname(output_path)):
            try:
                os.makedirs(os.path.dirname(output_path))
            except:
                pass
        output = pd.HDFStore(output_path, mode='w', complib='zlib',
                             complevel=1, fletcher32=True)

        def put_data(j, group):
            analysis_df = pd.DataFrame(out_xhat[j, :, :], index=self.times)
            analysis_err_df = pd.DataFrame(out_phat[j, :, :], index=self.times)
            analysis_at_sensors = pd.DataFrame(xhat_at_sens[j, :, :],
                                               index=self.times,
                                               columns=self.sensor_lats.index)
            orig_at_sensors = pd.DataFrame(back_at_sens[j, :, :],
                                           index=self.times,
                                           columns=self.sensor_lats.index)
            trace_df = pd.DataFrame(out_trHPH[j], index=self.times,
                                    columns=['trHPH', 'trHPH_wh', 'trHPaH',
                                             'trHPaH_wh', 'trR', 'trR_wh'])
            heights = pd.Series(out_Hc[j], index=self.times)

            output.put('{}/analysis'.format(group), analysis_df)
            output.put('{}/analysis_error'.format(group), analysis_err_df)
            output.put('{}/sensor_analysis'.format(group), analysis_at_sensors)
            output.put('{}/sensor_background'.format(group),  orig_at_sensors)
            output.put('{}/error_traces'.format(group), trace_df)
            output.put('{}/adjusted_heights'.format(group), heights)
            mse = ((analysis_at_sensors.ix[self.times, self.analyze_sensors] -
                    self.sensor_data)**2).mean().mean()
            return mse

        if self.shift_clouds:
            shifted_mse = put_data(0, 'shifted')
        else:
            shifted_mse = None

        if self.no_shift:
            unshifted_mse = put_data(1, 'unshifted')
        else:
            unshifted_mse = None

        output.put('clear_times', self.clear_times.to_series())
        output.put('cloudy_times', self.cloudy_times.to_series())
        output._handle.root._v_attrs.input_md5sum = self.dataset_md5
        output.flush()
        output.close()
        return shifted_mse, unshifted_mse


def main(args=None):
    sys.excepthook = handle_exception
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args is None:
        args = parse_args()

    CO = CalcOI(args)
    CO.load_data()
    CO.compute_Hs()
    CO.make_outputs(2)
    return CO.run()


if __name__ == '__main__':
    main()
