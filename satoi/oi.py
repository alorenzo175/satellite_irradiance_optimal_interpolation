from string import Template
import operator
import logging

import numpy as np
import pandas as pd
import pvlib
from scipy import spatial, linalg, optimize


class OI(object):
    _pairwised = None
    _pairwised_dtype = None
    _squared_exp_kernel = None
    _squared_exp_kernel_dtype = None
    _gamma_exp_kernel = None
    _gamma_exp_kernel_dtype = None
    _atan2func = None
    _atan2_dtype = None

    def __init__(self, arch='cpu', gpu_context=None, dtype='float32'):
        if arch == 'gpu':
            global pycuda
            import pycuda.tools
            global gpuarray
            import pycuda.gpuarray as gpuarray
            global cumath
            import pycuda.cumath as cumath
            global ElementwiseKernel
            from pycuda.elementwise import ElementwiseKernel
            global SourceModule
            from pycuda.compiler import SourceModule
            global culinalg
            import skcuda.linalg as culinalg
            global misc
            import skcuda.misc as misc
            if gpu_context is None:
                self.context = pycuda.tools.make_default_context()
            else:
                self.context = gpu_context
            self.device = self.context.get_device()
            culinalg.init()

        self.arch = arch

    def sphere_to_lcc(self, lats, lons, R=6370, truelat0=31.7, truelat1=31.7,
                      ref_lat=31.68858, stand_lon=-113.7):
        """
        Convert from spherical lats/lons like what comes out of WRF to the WRF
        Lambert Conformal x/y coordinates. Defaults are what
        are generally used for the AZ domain
        """
        phis = np.radians(lats)
        lambdas = np.radians(lons)
        phi0 = np.radians(ref_lat)
        phi1 = np.radians(truelat0)
        phi2 = np.radians(truelat1)
        lambda0 = np.radians(stand_lon)

        if truelat0 == truelat1:
            n = np.sin(phi0)
        else:
            n = (np.log(np.cos(phi1) / np.cos(phi2)) /
                 np.log(np.tan(np.pi / 4 + phi2 / 2) /
                        np.tan(np.pi / 4 + phi1 / 2)
                        ))
        F = (np.cos(phi1) * np.power(np.tan(np.pi / 4 + phi1 / 2), n) / n)
        rho0 = F / np.power(np.tan(np.pi / 4 + phi0 / 2), n)
        rho = F / np.power(np.tan(np.pi / 4 + phis / 2), n)
        x = R * rho * np.sin(n * (lambdas - lambda0))
        y = R * (rho0 - rho * np.cos(n * (lambdas - lambda0)))

        return x, y

    def lcc_to_sphere(self, x, y, R=6370, truelat0=31.7, truelat1=31.7,
                      ref_lat=31.68858, stand_lon=-113.7):
        """
        Convert from spherical lats/lons like what comes out of WRF to the WRF
        Lambert Conformal x/y coordinates. Defaults are what
        are generally used for the AZ domain
        """
        phi0 = np.radians(ref_lat)
        phi1 = np.radians(truelat0)
        phi2 = np.radians(truelat1)
        lambda0 = np.radians(stand_lon)

        if truelat0 == truelat1:
            n = np.sin(phi0)
        else:
            n = (np.log(np.cos(phi1) / np.cos(phi2)) /
                 np.log(np.tan(np.pi / 4 + phi2 / 2) /
                        np.tan(np.pi / 4 + phi1 / 2)
                        ))
        F = (np.cos(phi1) * np.power(np.tan(np.pi / 4 + phi1 / 2), n) / n)
        rho0 = F / np.power(np.tan(np.pi / 4 + phi0 / 2), n)
        x = x / R
        y = y /R
        rho = np.sqrt(x**2 + (y - rho0)**2)
        phis = 2 * (np.arctan2(F**(1.0 / n), rho**(1.0 / n)) - np.pi / 4)
        lambdas = np.arcsin(x / rho) / n + lambda0

        return np.degrees(phis), np.degrees(lambdas)

    def lcc_to_sphere_cuda(self, x, y, R=6370, truelat0=31.7, truelat1=31.7,
                           ref_lat=31.68858, stand_lon=-113.7):
        phi0 = np.radians(ref_lat)
        phi1 = np.radians(truelat0)
        phi2 = np.radians(truelat1)
        lambda0 = np.radians(stand_lon)

        if truelat0 == truelat1:
            n = np.sin(phi0)
        else:
            n = (np.log(np.cos(phi1) / np.cos(phi2)) /
                 np.log(np.tan(np.pi / 4 + phi2 / 2) /
                        np.tan(np.pi / 4 + phi1 / 2)
                        ))
        F = (np.cos(phi1) * np.power(np.tan(np.pi / 4 + phi1 / 2), n) / n)
        rho0 = F / np.power(np.tan(np.pi / 4 + phi0 / 2), n)
        x = x / R
        y = y / R
        ymrho = y - rho0
        rho = cumath.sqrt(x*x + ymrho * ymrho)
        atan1 = F**(1.0 / n)
        atan2 = rho**(1.0 / n)
        atan_res = self.atan2(atan1, atan2)
        phis = 360 * (atan_res - np.pi / 4) / np.pi
        lambdas = (cumath.asin(x / rho) / n + lambda0 ) * 180 / np.pi
        return phis, lambdas

    def atan2(self, a, b):
        if b.dtype.itemsize == 8:
            dtype = 'double'
        else:
            dtype = 'float'

        atan2_func = self._atan2(dtype)
        res = gpuarray.zeros(b.shape, b.dtype)
        atan2_func(a, b, res)
        return res

    def _atan2(self, dtype):
        if (self._atan2func is None and dtype != self._atan2_dtype):
            atan2 = ElementwiseKernel(
                    "{dtype} x, {dtype} *y, {dtype} *z".format(dtype=dtype),
                    "z[i] = atan2(x, y[i])",
                    "atan2_kern")
            self._atan2func = atan2
        return self._atan2func

    def distance_correlation(self, sat_lats, sat_lons, gamma, length):
        satx, saty = self.sphere_to_lcc(sat_lats.ravel(), sat_lons.ravel())
        satx = satx[:, None]
        saty = saty[:, None]

        dists = np.sqrt((satx - satx.T)**2 + (saty - saty.T)**2)
        if gamma == 0:
            C = (1 - (dists / length))
            C[C < 0] = 0
        else:
            C = np.exp(-1.0 * (dists / length)**gamma)
        return C

    def _make_tree(self, sat_lats, sat_lons):
        satx, saty = self.sphere_to_lcc(
            sat_lats.ravel(), sat_lons.ravel())
        self._tree = spatial.cKDTree(
            np.vstack((satx, saty)).T, compact_nodes=False,
            balanced_tree=False,
            leafsize=30)

    def compute_H(self, sensor_lats, sensor_lons, sat_lats, sat_lons, k=1):
        sensx, sensy = self.sphere_to_lcc(sensor_lats.ravel(),
                                          sensor_lons.ravel())
        if not hasattr(self, '_tree'):
            self._make_tree(sat_lats, sat_lons)

        d, inds = self._tree.query(np.vstack((sensx, sensy)).T, k)
        if k == 1:
            inds = inds.reshape((inds.shape[0], 1))
            d = d.reshape((d.shape[0], 1))

        H = np.zeros((len(sensor_lats), len(sat_lons.ravel())))
        for i in range(inds.shape[1]):
            w = 1.0 / d[:, i]
            wsum = (1.0 / d).sum(1)
            H[np.arange(len(sensor_lats)), inds[:, i]] = w / wsum
        return H

    def calc_nearby_points(self, sensor_lats, sensor_lons, sat_lats, sat_lons, r):
        sensx, sensy = self.sphere_to_lcc(sensor_lats.ravel(),
                                          sensor_lons.ravel())
        if not hasattr(self, '_tree'):
            self._make_tree(sat_lats, sat_lons)
        return self._tree.query_ball_point(np.vstack((sensx, sensy)).T, r)

    def pixel_similarity_cpu(self, image):
        image = image.ravel().reshape(image.shape[0] * image.shape[1], 1)
        norm_diff = np.abs(image - image.T)
        C = (1 - norm_diff / (norm_diff.max() - norm_diff.min()))
        return C

    def pairwised(self, N, dtype):
        if (self._pairwised is None or dtype != self._pairwised_dtype):
            func_mod_template = Template("""
            // Macro for converting subscripts to linear index:
            #define INDEX(a, b) a*${M}+b

            __global__ void func(${dtype} *x, ${dtype} *y, unsigned int N) {
            // Obtain the linear index corresponding to the current thread:
            unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                               blockIdx.x*${max_threads_per_block}+threadIdx.x;

            // Convert the linear index to subscripts:
            unsigned int a = idx/${M};
            unsigned int b = idx%${M};
            ${dtype} Pvalue = 0.0;

            // Use the subscripts to access the array:
            if (idx < N) {
                Pvalue = y[a] - y[b];
                if (Pvalue<0) {
                    Pvalue *= -1.0;
                }
                x[INDEX(a,b)] = Pvalue;
                }
            }
            """)
            max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(  # NOQA
                self.device)
            max_blocks_per_grid = max(max_grid_dim)
            func_mod = SourceModule(func_mod_template.substitute(
                max_threads_per_block=max_threads_per_block,
                max_blocks_per_grid=max_blocks_per_grid,
                M=N, dtype=dtype))

            self._pairwised = func_mod.get_function('func')
            self._pairwised_dtype = dtype

        return self._pairwised

    def pairwise_difference(self, in_gpu, N):
        out = gpuarray.empty((N, N), in_gpu.dtype)
        block_dim, grid_dim = misc.select_block_grid_sizes(
            self.device, (N, N))
        if in_gpu.dtype.itemsize == 8:
            dtype = 'double'
        else:
            dtype = 'float'
        pairwised = self.pairwised(N, dtype)
        pairwised(out.gpudata, in_gpu.gpudata, np.uint32(out.size),
                  block=block_dim, grid=grid_dim)
        return out

    def gdivide(self, x_gpu, y_gpu):
            return misc.binaryop_2d("/", operator.truediv, False, x_gpu, y_gpu)

    def pixel_similarity_cuda(self, image):
        N = image.shape[0]
        nd = self.pairwise_difference(image, N)
        diff = gpuarray.max(nd) - gpuarray.min(nd)
        norm = self.gdivide(nd, diff)
        C = 1 - norm
        return C

    def pixel_similarity(self, image):
        if self.arch == 'cpu':
            return self.pixel_similarity_cpu(image)
        elif self.arch == 'gpu':
            return self.pixel_similarity_cuda(image)
        else:
            raise ValueError('Arch must be cpu or gpu')

    def linear_corr_cpu(self, image, l):
        r = image.ravel()[:, None]
        diff = np.abs(r - r.T)
        C = (1 - (diff / l))
        C[C < 0] = 0
        return C

    def linear_corr_cuda(self, image, l):
        N = image.shape[0]
        nd = self.pairwise_difference(image, N)
        C = (1 - (nd / l))
        zeros = misc.zeros(C.shape, C.dtype)
        C = gpuarray.if_positive(C, C, zeros)
        return C.copy()

    def linear_corr(self, image, l):
        if self.arch == 'cpu':
            return self.linear_corr_cpu(image, l)
        elif self.arch == 'gpu':
            return self.linear_corr_cuda(image, l)
        else:
            raise ValueError('Arch must be cpu or gpu')

    def squared_exponential_correlation_cpu(self, image, l):
        """
        Compute the squared exponentional correlation function
        given by:
          C_{i,j} = exp(- d_{ij}^2/2l^2)
        where l is some characteristic length and d_{i,j} is the
        distance (not neccessarily spatial) between the two
        points, i and j
        """
        # image denoted v
        image_vector = image.ravel().reshape(image.shape[0] * image.shape[1], 1)
        # d_{i,j} = v_{i,j} - v_{j, i}
        d = image_vector - image_vector.T
        C = np.exp(- d**2 / (2 * l**2))
        return C

    def squared_exp_kernel(self, N, dtype):
        if (self._squared_exp_kernel is None or
                dtype != self._squared_exp_kernel_dtype):
            func_mod_template = Template("""
            // Macro for converting subscripts to linear index:
            #define INDEX(a, b) a*${M}+b

            __global__ void func(${dtype} *x, ${dtype} *y, ${dtype} *l, unsigned int N) {
            // Obtain the linear index corresponding to the current thread:
            unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                               blockIdx.x*${max_threads_per_block}+threadIdx.x;

            // Convert the linear index to subscripts:
            unsigned int a = idx/${M};
            unsigned int b = idx%${M};
            ${dtype} Pvalue = 0.0;

            // Use the subscripts to access the array:
            if (idx < N) {
                Pvalue = exp(-1.0*(y[a] - y[b])*(y[a] - y[b])/(2*l[0]*l[0]));
                x[INDEX(a,b)] = Pvalue;
                }
            }
            """)
            max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(  # NOQA
                self.device)
            max_blocks_per_grid = max(max_grid_dim)
            func_mod = SourceModule(func_mod_template.substitute(
                max_threads_per_block=max_threads_per_block,
                max_blocks_per_grid=max_blocks_per_grid,
                M=N, dtype=dtype))

            self._squared_exp_kernel = func_mod.get_function('func')
            self._squared_exp_kernel_dtype = dtype

        return self._squared_exp_kernel

    def squared_exponential_correlation_cuda(self, image, l):
        if isinstance(image, pycuda.gpuarray.GPUArray):
            N = image.shape[0]
            image_g = image
        else:
            N = image.shape[0] * image.shape[1]
            image = image.ravel().reshape(N, 1)
            image_g = gpuarray.to_gpu(image)
        l_g = gpuarray.to_gpu(np.array([l]).astype(image.dtype))
        C = gpuarray.empty((N, N), image_g.dtype)
        block_dim, grid_dim = misc.select_block_grid_sizes(
            self.device, (N, N))
        if image.dtype.itemsize == 8:
            dtype = 'double'
        else:
            dtype = 'float'
        kernel_func = self.squared_exp_kernel(N, dtype)
        kernel_func(C.gpudata, image_g.gpudata, l_g.gpudata,
                    np.uint64(C.size),
                    block=block_dim, grid=grid_dim)
        return C

    def squared_exponential_correlation(self, image, l):
        if self.arch == 'cpu':
            return self.squared_exponential_correlation_cpu(image, l)
        elif self.arch == 'gpu':
            return self.squared_exponential_correlation_cuda(image, l)
        else:
            raise ValueError('Arch must be cpu or gpu')

    def gamma_exponential_correlation_cpu(self, image, l, gamma):
        """
        Compute the gamma exponentional correlation function
        given by:
          C_{i,j} = exp(- (d_{ij}/l)^\gamma)
        where l is some characteristic length and d_{i,j} is the
        distance (not neccessarily spatial) between the two
        points, i and j, and gamma is the exponent
        """
        # image denoted v
        image_vector = image.ravel().reshape(image.shape[0] * image.shape[1], 1)
        # d_{i,j} = v_{i,j} - v_{j, i}
        d = np.abs(image_vector - image_vector.T)
        C = np.exp(- (d/l)**gamma)
        return C

    def gamma_exp_kernel(self, N, dtype):
        if (self._gamma_exp_kernel is None or
                dtype != self._gamma_exp_kernel_dtype):
            func_mod_template = Template("""
            // Macro for converting subscripts to linear index:
            #define INDEX(a, b) a*${M}+b

            __global__ void func(${dtype} *x, ${dtype} *y, ${dtype} *l, unsigned int N) {
            // Obtain the linear index corresponding to the current thread:
            unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                               blockIdx.x*${max_threads_per_block}+threadIdx.x;

            // Convert the linear index to subscripts:
            unsigned int a = idx/${M};
            unsigned int b = idx%${M};
            ${dtype} Pvalue = 0.0;
            ${dtype} darg = 0.0;
            ${dtype} abs_arg = 0.0;

            // Use the subscripts to access the array:
            if (idx < N) {
                darg = (y[a] - y[b]) / l[0];
                abs_arg = fabs(darg);
                Pvalue = exp(-1.0*pow(abs_arg, l[1]));
                x[INDEX(a,b)] = Pvalue;
                }
            }
            """)
            max_threads_per_block, max_block_dim, max_grid_dim = misc.get_dev_attrs(  # NOQA
                self.device)
            max_blocks_per_grid = max(max_grid_dim)
            func_mod = SourceModule(func_mod_template.substitute(
                max_threads_per_block=max_threads_per_block,
                max_blocks_per_grid=max_blocks_per_grid,
                M=N, dtype=dtype))

            self._gamma_exp_kernel = func_mod.get_function('func')
            self._gamma_exp_kernel_dtype = dtype

        return self._gamma_exp_kernel

    def gamma_exponential_correlation_cuda(self, image, l, y):
        if isinstance(image, pycuda.gpuarray.GPUArray):
            N = image.shape[0]
            image_g = image
        else:
            N = image.shape[0] * image.shape[1]
            image = image.ravel().reshape(N, 1)
            image_g = gpuarray.to_gpu(image)
        params = gpuarray.to_gpu(np.array([l, y]).astype(image.dtype))
        C = gpuarray.empty((N, N), image_g.dtype)
        block_dim, grid_dim = misc.select_block_grid_sizes(
            self.device, (N, N))
        if image.dtype.itemsize == 8:
            dtype = 'double'
        else:
            dtype = 'float'
        kernel_func = self.gamma_exp_kernel(N, dtype)
        kernel_func(C.gpudata, image_g.gpudata, params.gpudata,
                    np.uint64(C.size),
                    block=block_dim, grid=grid_dim)
        return C.copy()

    def gamma_exponential_correlation(self, image, l, y):
        if self.arch == 'cpu':
            return self.gamma_exponential_correlation_cpu(image, l, y)
        elif self.arch == 'gpu':
            return self.gamma_exponential_correlation_cuda(image, l, y)
        else:
            raise ValueError('Arch must be cpu or gpu')

    def compute_P(self, C, D):
        if self.arch == 'cpu':
            return self.compute_P_cpu(C, D)
        elif self.arch == 'gpu':
            return self.compute_P_cuda(C, D)
        else:
            raise ValueError('Arch must be cpu or gpu')

    def compute_P_cpu(self, C, D):
        P = D.dot(C.dot(D))
        return P

    def compute_P_cuda(self, C, D):
        dD = culinalg.diag(D)
        CD = culinalg.dot_diag(dD, C, 'T')
        P = culinalg.dot_diag(dD, CD)
        return P.copy()

    def compute_HPH(self, P, H, HT):
        if self.arch == 'cpu':
            return self.compute_HPH_cpu(P, H, HT)
        elif self.arch == 'gpu':
            return self.compute_HPH_cuda(P, H, HT)
        else:
            raise ValueError('Arch must be cpu or gpu')

    def compute_HPH_cpu(self, P, H, HT):
        hph = H.dot(P.dot(HT))
        return hph

    def compute_HPH_cuda(self, P, H, HT):
        PHT = culinalg.dot(P, HT)
        hph = culinalg.dot(H, PHT)
        return hph.copy()

    def compute_analysis(self, xb, y, R, P, H, HT=None, HPH=None, calcP=True):
        if self.arch == 'cpu':
            return self.compute_analysis_cpu(xb, y, R, P, H, HT, HPH, calcP)
        elif self.arch == 'gpu':
            return self.compute_analysis_cuda2(xb, y, R, P, H, HT, HPH, calcP)
        else:
            raise ValueError('Arch must be cpu or gpu')

    def compute_analysis_cpu(self, xb, y, R, P, H, HT=None, hph=None,
                             calcP=True):
        if HT is None:
            HT = H.T
        if hph is None:
            hph = self.compute_HPH(P, H, HT)

        inv = linalg.inv(R + hph)
        W = P.dot(HT.dot(inv))
        xhat = xb + W.dot(y - H.dot(xb))
        if calcP:
            Phat = (np.eye(P.shape[0]) - W.dot(H)).dot(P)
        else:
            Phat = np.zeros((1,))
        return xhat, Phat

    def compute_analysis_cuda(self, xb, y, R, P, H, HT=None, hph=None,
                              calcP=True):
        if HT is None:
            HT = culinalg.transpose(H)
        if hph is None:
            hph = self.compute_HPH(P, H, HT)
        Rhph = misc.add(R, hph)
        inv = culinalg.inv(Rhph)
        HTinv = culinalg.dot(HT, inv)
        W = culinalg.dot(P, HTinv)
        #W = culinalg.dot(P, culinalg.dot(HT, inv))
        Hxb = culinalg.dot(H, xb)
        yHxb = misc.subtract(y, Hxb)
        WyHxb = culinalg.dot(W, yHxb)
        xhat = misc.add(xb, WyHxb)
        #xhat = xb + culinalg.dot(W, (y - culinalg.dot(H, xb)))
        if calcP:
            I = culinalg.eye(P.shape[0])
            WH = culinalg.dot(W, H)
            IWH = I - WH
            Phat = culinalg.dot(IWH, P)
        else:
            Phat = misc.zeros((1,), dtype=P.dtype)
        return xhat, Phat

    def compute_analysis_cuda2(self, xb, y, R, P, H, HT=None, hph=None,
                              calcP=True):
        if HT is None:
            HT = culinalg.transpose(H)
        HP = culinalg.dot(H, P)
        if hph is None:
            hph = culinalg.dot(HP, HT)
        Rhph = misc.add(R, hph)
        inv = culinalg.inv(Rhph)
        W = culinalg.dot(HP, inv, transa='T')
        Hxb = culinalg.dot(H, xb)
        yHxb = misc.subtract(y, Hxb)
        WyHxb = culinalg.dot(W, yHxb)
        xhat = misc.add(xb, WyHxb)
        #xhat = xb + culinalg.dot(W, (y - culinalg.dot(H, xb)))
        if calcP:
            I = culinalg.eye(P.shape[0])
            WH = culinalg.dot(W, H)
            IWH = I - WH
            Phat = culinalg.dot(IWH, P)
        else:
            Phat = misc.zeros((1,), dtype=P.dtype)
        return xhat, Phat

    def adjust_locations(self, time, sat_lats, sat_lons,
                         Hc, other=None, sensors=True):
        if other is not None:
            solpos, sat_x, sat_y, img_x, img_y = other
        else:
            try:
                time = pd.Timestamp(time).tz_localize('MST')
            except TypeError:
                time = pd.Timestamp(time).tz_convert('MST')
            solpos = pvlib.solarposition.get_solarposition(
                time, 32.2, -110.95).iloc[0]
            sat_x, sat_y = self.sphere_to_lcc(0, -135)
            img_x, img_y = self.sphere_to_lcc(sat_lats, sat_lons)

        tan_sat_elevation = 35786 / np.sqrt((sat_x - img_x)**2 +
                                            (sat_y - img_y)**2)
        shift_angle = np.arctan2((img_y - sat_y), (img_x - sat_x))
        shift_length = Hc / tan_sat_elevation
        x_adj = shift_length * np.cos(shift_angle)
        y_adj = shift_length * np.sin(shift_angle)
        sol_shift_length = Hc / np.tan(np.radians(solpos['elevation']))
        x_sun = sol_shift_length * np.sin(np.radians(solpos['azimuth']))
        y_sun = sol_shift_length * np.cos(np.radians(solpos['azimuth']))
        if sensors:
            adj_sat_lats, adj_sat_lons = self.lcc_to_sphere(
                img_x + x_adj + x_sun, img_y + y_adj + y_sun)
        else:
            adj_sat_lats, adj_sat_lons = self.lcc_to_sphere(
                img_x - x_adj - x_sun, img_y - y_adj - y_sun)
        return adj_sat_lats, adj_sat_lons

    def analysis_mse_minimizer(self, Hc, time, sat_lats, sat_lons, sensor_lats,
                               sensor_lons, dropped_sensor_lats,
                               dropped_sensor_lons, cloud_radius, k,
                               xb, y, R, P, other=None, other2=None,
                               dtype='float32',
                               retH=False):
        if isinstance(sensor_lats, pd.Series):
            slat = sensor_lats.values
            slon = sensor_lons.values
        else:
            slat = sensor_lats
            slon = sensor_lons

        adj_sen_lats, adj_sen_lons = self.adjust_locations(
            time, slat, slon, Hc, other=other)
        Hp = self.compute_H(adj_sen_lats, adj_sen_lons, sat_lats,
                            sat_lons, k)
        if isinstance(dropped_sensor_lats, pd.Series):
            dslat = dropped_sensor_lats.values
            dslon = dropped_sensor_lons.values
        else:
            dslat = dropped_sensor_lats
            dslon = dropped_sensor_lons

        adj_sen_lats, adj_sen_lons = self.adjust_locations(
            time, dslat, dslon, Hc, other=other2)
        H = self.compute_H(adj_sen_lats, adj_sen_lons,
                           sat_lats, sat_lons, k)

        if hasattr(self, 'xbng'):
                xb_nogpu = self.xbng
        else:
            if self.arch == 'gpu':
                xb_nogpu = xb.get().copy()
            else:
                xb_nogpu = xb.copy()

        oH = H.copy()
        if cloud_radius is not None:
            nearby_inds = self.calc_nearby_points(adj_sen_lats, adj_sen_lons,
                                                  sat_lats, sat_lons,
                                                  cloud_radius)
            remove = []
            for i in range(nearby_inds.shape[0]):
                nby_pts = xb_nogpu[nearby_inds[i]]
                if nby_pts.shape[0] == 0:
                    max_diff = 1
                else:
                    max_diff = nby_pts.max() - nby_pts.min()

                if max_diff >= 0.2:
                    remove.append(i)
                    H[i] = np.zeros(H.shape[1])
                    #if self.arch == 'gpu':
                    #    cmeas[i] = gpuarray.zeros(1, dtype=dtype)
                    #else:
                    #    cmeas[i] = np.zeros(1)

        HT = H.T.copy()
        if self.arch == 'gpu':
            H = gpuarray.to_gpu(H).astype(dtype)
            Hp = gpuarray.to_gpu(Hp).astype(dtype)
            HT = gpuarray.to_gpu(HT).astype(dtype)
            oH = gpuarray.to_gpu(oH).astype(dtype)

        if retH:
            return H, HT, Hp
        xhat, Phat = self.compute_analysis(xb, y, R, P, H, HT, None, False)


        if hasattr(self, 'cmeas'):
            cmeas = self.cmeas
        else:
            cmeas = y.copy()

        if self.arch == 'gpu':
            mse = ((culinalg.dot(oH, xhat) - cmeas)**2).get().mean()
        else:
            mse = np.mean((oH.dot(xhat) - cmeas)**2)

        return np.sqrt(mse)

    def minimize_analysis_mse(self, time, sat_lats, sat_lons, sensor_lats,
                              sensor_lons, dropped_sensor_lats,
                              dropped_sensor_lons, k, cloud_radius,
                              xb, y, R, P, dtype='float32'):
        """
        Computes H and Hp by shifting the image to minimize the analysis RMSE
        """
        try:
            time = pd.Timestamp(time).tz_localize('MST')
        except TypeError:
            time = pd.Timestamp(time).tz_convert('MST')

        solpos = pvlib.solarposition.get_solarposition(
            time, sat_lats.ravel().mean(), sat_lons.ravel().mean()).iloc[0]
        sat_x, sat_y = self.sphere_to_lcc(0, -135)
        img_x, img_y = self.sphere_to_lcc(sensor_lats, sensor_lons)
        if isinstance(img_x, pd.Series):
            img_x = img_x.values
            img_y = img_y.values
        other = (solpos, sat_x, sat_y, img_x, img_y)

        img_x, img_y = self.sphere_to_lcc(dropped_sensor_lats,
                                          dropped_sensor_lons)
        if isinstance(img_x, pd.Series):
            img_x = img_x.values
            img_y = img_y.values
        other2 = (solpos, sat_x, sat_y, img_x, img_y)

        args = [time, sat_lats, sat_lons, sensor_lats, sensor_lons,
                dropped_sensor_lats, dropped_sensor_lons,
                cloud_radius, k, xb, y, R, P, other, other2, dtype]
        if self.arch == 'gpu':
            self.xbng = xb.get().copy()
        else:
            self.xbng = xb.copy()
        self.cmeas = y.copy()
        res = optimize.brute(self.analysis_mse_minimizer, Ns=51,
                             args=tuple(args), ranges=((0, 14),))

        del self.xbng
        del self.cmeas
        Hc = res[0]
        nargs = args.copy()
        nargs.insert(0, Hc)
        nargs.append('True')
        H, HT, Hp = self.analysis_mse_minimizer(*nargs)
        return H, HT, Hp, Hc


def clear_times(sat_ci, obs_ci):
    def max_dev(series):
        s = (1.0 - series).sort_values(ascending=False)
        return s.iloc[1]

    return ((sat_ci.min(axis=1) >= 0.8) &
            (sat_ci.mean(axis=1) >= 0.99) &
            (obs_ci.apply(max_dev, axis=1) <= 0.05))
