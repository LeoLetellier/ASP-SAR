#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""\
ts.py
-------------
Temporal decomposition of the time series delays of selected pixels 
(used depl_cumule (BIP format) and images_retenues, output of invers_pixel).

Usage: 
    ts.py (<cols> <lines> | <region>) [--cube=<cube> --list_images=<list_images> --windowsize=<windowsize> \
--windowrefsize=<windowrefsize> --lectfile=<lectfile> --rms=<rms> --linear --steps=<steps> --postseismic=<postseismic> \
--seasonal --seasonal_increase [--vector=<vector>] [--info=<info>] --semiannual --biannual --bperp --imref=<imref> \
--cond=<cond> --slowslip=<slowslip> --ineq=<ineq> --name=<name> --scale=<scale> --plot <iref> <jref> \
--bounds=<bounds> --dateslim=<dateslim> --plot_dateslim=<plot_dateslim> --color=<color> --fillstyle=<fillstyle>]


invers_disp_pixel.py -h | --help

-h --help               Show this screen
"""


import numpy as np
import scipy.optimize as opt
import numpy.linalg as lst
import os
import docopt
from copy import deepcopy
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
gdal.UseExceptions()


class PatternOptions:
    def __init__(self):
        self.reference = ReferencePattern()
        self.linear = None
        self.seasonal = None
        self.seasonalt = None
        self.semiannual = None
        self.biannual = None
        self.steps = []
        self.postseismic = []
        self.slowslip = []
        self.bperp = None
        self.vector = []
        self.degreeday = None
    
    def as_list(self):
        list_pattern = [self.reference]

        def append_if_exists(var, list_p):
            if bool(var):
                if type(var) is not list:
                    var = [var]
                list_p += var
            return list_p
        
        list_src = [self.linear, self.seasonal, self.seasonalt, self.semiannual, self.biannual,
        self.steps, self.postseismic, self.slowslip, self.bperp, self.vector, self.degreeday]
        for s in list_src:
            append_if_exists(s, list_pattern)
        
        return list_pattern
    
    def ids(self):
        ids = [0]
        current_id = 1

        def update_id(ids, var):
            if bool(var):
                if type(var) is not list:
                    var = [var]
                ids.append(current_id)
                current_id += len(var)
            else:
                ids.append(None)
                current_id += 1
        
        list_src = [self.linear, self.seasonal, self.seasonalt, self.semiannual, self.biannual,
        self.steps, self.postseismic, self.slowslip, self.bperp, self.vector, self.degreeday]
        for s in list_src:
            update_id(ids, s)
        
        return ids


class TimeModel:
    def __init__(self, time_patterns, dates_dec):
        self.dates = dates_dec
        self.patterns_opts = time_patterns
        self.patterns = time_patterns.as_list()
        self.matrix = self._build_matrix()
        self.coeffs = np.ones(shape=len(self.patterns))
        self.sigmas = np.ones(shape=len(self.patterns))

    def _build_matrix(self):
        vecs = [s(self.dates) for s in self.patterns]
        return np.column_stack(vecs)

    def __str__(self):
        msg = "=> Time model\n"
        msg += "\t> Dates: {} > {}\n".format(self.dates[0], self.dates[-1])
        for i, p in enumerate(self.patterns):
            msg += "\t> {}\n".format(p)
            msg += "\tcoeff: {}\n".format(self.coeffs[i])
            msg += "\tsigma: {}\n".format(self.sigmas[i])
            msg += "\t________\n"
        return msg

    def copy(self):
        return deepcopy(self)

    def __call__(self, time, reference_date=None):
        if reference_date is None:
            return np.sum([self.coeffs[k] * self.patterns[k](time) for k in len(self.patterns)], axis=0)
        return np.sum([self.coeffs[k] * self.patterns[k](time - reference_date) for k in len(self.patterns)], axis=0)
    
    def reconstruct(self):
        return np.dot(self.matrix, self.coeffs), self.sigmas
    
    @staticmethod
    def from_params(base_date, linear=False, seasonal=False, seasonalt=False, semiannual=False, biannual=False, steps=None, postseismic=None, slowslip=None, vector=None, bperp=None, degreeday=None):
        pattern_opts = PatternOptions()
        if linear:
            pattern_opts.linear = LinearPattern()
        if seasonal:
            pattern_opts.seasonal = [SinVarPattern(), CosVarPattern()]
        if seasonalt:
            pattern_opts.seasonalt = [SinTPattern(), CosTPattern()]
        if semiannual:
            pattern_opts.semiannual = [SinVarPattern(4 * np.pi), CosVarPattern(4 * np.pi)]
        if biannual:
            pattern_opts.biannual = [SinVarPattern(np.pi), CosVarPattern(np.pi)]
        if steps is not None:
            for s in steps:
                pattern_opts.steps.append(StepPattern(s))
        if postseismic is not None:
            for ps in postseismic:
                pattern_opts.postseismic.append(PostSeismicPattern(ps))
        if slowslip is not None:
            for ss in slowslip:
                pattern_opts.slowslip.append(SlowSlipPattern(ss))
        if bperp is not None:
            pattern_opts.bperp = BPerpPattern(bperp, base_date)
        if vector is not None:
            for v in vector:
                pattern_opts.vector.append(ValuePattern(v, base_date))
        if degreeday is not None:
            pattern_opts.degreeday = StephanPattern(degreeday[0], degreeday[1])
        
        return TimeModel(pattern_opts, base_date)

    def solve(self, values, sigma=None, **kwargs):
        if len(values) != len(self.dates):
            raise ValueError("Number of values differs from the number of dates in the model ({}!={})".format(len(values), len(self.dates)))

        pixel_date_valid = np.where(np.isnan(values), False, True)
    
        data = values[pixel_date_valid]
        model = self.matrix[pixel_date_valid, :]
        dates_valid = self.dates[pixel_date_valid]

        if sigma is None:
            sigma = np.ones(len(dates_valid))
        else:
            if len(sigma) != len(self.dates):
                raise ValueError("Number of sigma differs from the number of dates in the model ({}!={})".format(len(sigma), len(self.dates)))
            sigma = sigma[pixel_date_valid]
        
        self.coeffs, self.sigmas = consInvert(model, data, sigma)
        return self
    
    def plot(self, true_date, values, values_rms=None, band=None, position=None, **kwargs):
        model, sigma = self.reconstruct()
        mpl_date = [datetime.strptime(d, "%Y%m%d") for d in true_date]

        # plot last band of cube and localization of actual TS
        if band is not None and position is not None:
            plt.figure()
            vmin, vmax = np.nanpercentile(band, (2, 98))
            # mid = vmax - vmin
            # vmin, vmax = mid - 2 * (mid - vmin),mid + 2 * (vmax - mid)
            plt.imshow(band, vmin=vmin, vmax=vmax)
            plt.scatter(position[0], position[1], marker='x', color='black', s=150.0)
            plt.colorbar()

        plt.figure()
        plt.scatter(mpl_date, values)
        plt.plot(mpl_date, model, color='black')
        plt.xlabel("Date")
        plt.ylabel("Displacement [unit]")
        plt.title("Data VS Model")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))

        # plot bperp if needed, full time + season

        # plot vectors if needed

        # plot model, error bars ? season ?
        # model
        # model - vectors
        # model - vectors - linear
        # model - vectors - linear - seasonals
        pass

class TimePattern:
    def __init__(self):
        self.name = None

    def __call__(self, time):
        pass

    def __str__(self):
        return self.name

    def plot(self):
        return None


class ReferencePattern(TimePattern):
    def __init__(self):
        super().__init__()
        self.name = "reference"
    
    def __call__(self, time):
        return np.ones(shape=time.shape)
    

class LinearPattern(TimePattern):
    def __init__(self):
        super().__init__()
        self.name = "linear"
    
    def __call__(self, time):
        return time


class StepPattern(TimePattern):
    def __init__(self, step):
        super().__init__()
        self.name = "step"
        self.step = step
    
    def __call__(self, time):
        return np.array(time > self.step, dtype=float)


class SinTPattern(TimePattern):
    def __init__(self, delay=2 * np.pi):
        super().__init__()
        self.name = "sin t"
        self.delay = delay
    
    def __call__(self, time):
        return time * np.sin(self.delay * time)


class CosTPattern(TimePattern):
    def __init__(self, delay=2 * np.pi):
        super().__init__()
        self.name = "cos t"
        self.delay = delay
    
    def __call__(self, time):
        return time * np.cos(self.delay * time)


class SinVarPattern(TimePattern):
    def __init__(self, delay=2 * np.pi):
        super().__init__()
        self.name = "sin"
        self.delay = delay
    
    def __call__(self, time):
        return np.sin(self.delay * time)


class CosVarPattern(TimePattern):
    def __init__(self, delay=2 * np.pi):
        super().__init__()
        self.name = "cos"
        self.delay = delay
    
    def __call__(self, time):
        return np.cos(self.delay * time)


class SlowSlipPattern(TimePattern):
    def __init__(self, tcar):
        super().__init__()
        self.name = "slow slip"
        self.tcar = tcar

    def __call__(self, time):
        return 0.5 * (np.tanh(time / self.tcar) - 1) + 1


class PostSeismicPattern(TimePattern):
    def __init__(self, tcar):
        super().__init__()
        self.name = "post seismic"
        self.tcar = tcar

    def __call__(self, time):
        return np.where(time < 0, 0, np.log10(1 + time / self.tcar))


class ValuePattern(TimePattern):
    def __init__(self, value, time):
        super().__init__()
        self.name = "b perp"
        self.value = value
        self.time = time
    
    def __call__(self, time):
        return np.interp(time, self.time, self.value)


class BPerpPattern(ValuePattern):
    def __init__(self, value, time):
        super().__init__(value, time)
        self.name = "bperp"


class StephanPattern(TimePattern):
    def __init__(self, t_t, t_f):
        super().__init__()
        self.name = "degreeday"
        self.t_thaw = t_t
        self.t_freeze = t_f
    
    def __call__(self, time):
        day = np.mod(time, 1)
        dt = self.t_freeze - self.t_thaw
        dd_t = np.where((day > self.t_thaw) & (day < self.t_freeze), day - self.t_thaw, dt)
        dd_f = np.zeros(len(time))
        dd_f[day > self.t_freeze] = day[day > self.t_freeze] - self.t_freeze
        dd_f[day < self.t_thaw] = day[day < self.t_thaw] + 1 - self.t_thaw

        return np.sqrt(dd_t) - np.sqrt(dt / (1 - dt)) * np.sqrt(dd_f)


def construct_pattern_list(arguments):
    list_pattern = [ReferencePattern()]

    if arguments.get("--linear", False):
        list_pattern.append(LinearPattern())
    if arguments.get("--steps", False):
        steps = arguments["--steps"].split(",")
        for s in steps:
            list_pattern.append(StepPattern(float(s)))
    if arguments.get("--seasonal", False):
        list_pattern.append(SinVarPattern())
        list_pattern.append(CosVarPattern())
    if arguments.get("--seasonalalt", False):
        list_pattern.append()
    
    return list_pattern


def read_list_images(list_images):
    dates, dates_dec, bperp = np.loadtxt(list_images, comments='#', usecols=(1,3,5), unpack=True)
    return dates.astype(int), dates_dec, bperp


def read_lectfile(lectfile, ndates=None):
    ncol, nlign = list(map(int, open(lectfile).readline().split(None, 2)[0:2]))
    if ndates is None:
        try:
            ndates = int(open(lectfile).readlines(4)[-1])
        except:
            ndates = 1
    return ncol, nlign, ndates


def generate_header(raster, dims):
    content = """ENVI
samples =   {}
lines =   {}
bands =  {}
header offset = 0
data type = 4
interleave = bip
byte order = 0""".format(dims[0], dims[1], dims[2])
    with open(os.path.splitext(raster)[0] + ".hdr", 'w') as headerfile:
        headerfile.write(content)


def ensure_gdal_header(file, lectfile=None, ndates=None):
    try:
        gdal.Open(file)
    except:
        if lectfile is None:
            lectfile = "lect.in"
        if os.path.isfile(lectfile):
            dims = read_lectfile(lectfile, ndates)
            generate_header(file, dims)
        else:
            raise ValueError('Cannot read raster:', file)


def open_gdal(file, band=1, crop=None, supp_ndv=None):
    ds = gdal.Open(file)

    if crop is None:
        crop = [0, ds.RasterXSize, 0, ds.RasterYSize]
    if band is None:
        data = ds.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])
        ndv = ds.GetRasterBand(1).GetNoDataValue()
    else:
        if band == -1:
            band = ds.RasterCount
        band = ds.GetRasterBand(band)
        ndv = band.GetNoDataValue()
        data = band.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])

    if data.dtype in [np.float32, np.float64]:
        if ndv is not None and ndv != np.nan:
            data[data==ndv] = np.nan
        if supp_ndv is not None and supp_ndv != np.nan:
            data[data==supp_ndv] = np.nan

    return data


def pixel_selection(cube, lectfile, ndates, arguments):
    if arguments.get("<region>", None) is not None:
        mask_region = arguments["<region>"]
        ensure_gdal_header(mask_region, lectfile=lectfile, ndates=ndates)
        mask = open_gdal(mask_region)
        min_row, max_row = np.where(np.any(mask, axis=1))[0][[0, -1]]
        min_col, max_col = np.where(np.any(mask, axis=0))[0][[0, -1]]
        print(min_row, min_col, max_row, max_col)
        print(mask.shape)
        mask = mask[min_row:max_row, min_col:max_col]
        # find crop that contains the mask
        # open each band one by one on this crop
        # when mask is 0 replace by np.nan
        # compute median with nanmedian
        crop = [min_col, max_col, min_row, max_row]
        first_band = open_gdal(cube, band=1, crop=crop)
        loaded_data = np.zeros(shape=(ndates, first_band.shape[0], first_band.shape[1]))
        loaded_data[0] = first_band
        for b in range(2, ndates + 1):
            loaded_data[b-1] = open_gdal(cube, band=b, crop=crop)
        loaded_data = loaded_data[mask.astype(bool)]
        data = np.transpose(loaded_data, (1,2,0)).reshape(loaded_data.shape[1] * loaded_data.shape[2], loaded_data.shape[0])
        data_med = np.nanmedian(data, axis=0)
    else:
        ncols = [int(c) for c in arguments["<cols>"].split(',')]
        nlines = [int(l) for l in arguments["<lines>"].split(',')]
        wdw = arguments.get('--windowsize', None)
        wdw = 0 if wdw is None else int(wdw)
        points = []
        for p in range(len(ncols)):
            crop = [ncols[p] - wdw, ncols[p] + wdw + 1, nlines[p] - wdw, nlines[p] + wdw + 1]

            # print(open_gdal(cube, None, crop))
            data = open_gdal(cube, None, crop)
            data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
            # print(data.shape, data)
            # first_band = open_gdal(cube, band=1, crop=crop)
            # loaded_data = np.zeros(shape=(ndates, first_band.shape[0], first_band.shape[1]))
            # loaded_data[0] = first_band
            # for b in range(2, ndates + 1):
            #     loaded_data[b-1] = open_gdal(cube, band=b, crop=crop)
            # data = np.transpose(loaded_data, (1,2,0)).reshape(loaded_data.shape[1] * loaded_data.shape[2], loaded_data.shape[0])
            data_med = [np.nanmedian(data, axis=1), np.nanquantile(data, 0.1, axis=1), np.nanquantile(data, 0.25, axis=1), np.nanquantile(data, 0.75, axis=1), np.nanquantile(data, 0.90, axis=1)]
            # print(data_med.shape, data_med)
            points.append(data_med)
    # print(points)
    return points


def ts_referencing(ts_list, ndates, arguments):
    ncols = int(arguments["<iref>"])
    nlines = int(arguments["<jref>"])
    wdw = int(arguments.get('--windowrefsize', 0))
    crop = [ncols - wdw, ncols + wdw + 1, nlines - wdw, nlines + wdw + 1]
    data = open_gdal(cube, None, crop)
    # first_band = open_gdal(cube, band=1, crop=crop)
    # loaded_data = np.zeros(shape=(ndates, first_band.shape[0], first_band.shape[1]))
    # loaded_data[0] = first_band
    # for b in range(2, ndates + 1):
    #     loaded_data[b-1] = open_gdal(cube, band=b, crop=crop)
    # ref = np.transpose(loaded_data, (1,2,0)).reshape(loaded_data.shape[1] * loaded_data.shape[2], loaded_data.shape[0])
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    ref = np.nanmedian(data, axis=1)

    ref_ts_list = []
    for t in ts_list:
        ref_ts_list.append(t - ref)
    return ref_ts_list


def consInvert(A,b,sigmad,ineq='yes',cond=1.0e-3, iter=100,acc=1e-6, eguality=False):
    '''Solves the constrained inversion problem.

    Minimize:

    ||Ax-b||^2

    Subject to:
    mmin < m < mmax
    '''

    if A.shape[0] != len(b):
        raise ValueError('Incompatible dimensions for A and b')

    if ineq == 'no':
        try:
          Cd = np.diag(sigmad**2, k = 0)
          fsoln = np.dot(np.linalg.inv(np.dot(np.dot(A.T,np.linalg.inv(Cd)),A)),np.dot(np.dot(A.T,np.linalg.inv(Cd)),b))
        except:
          fsoln = lst.lstsq(A,b,rcond=None)[0]
    else:
        # if len(indexpo>0):
        #   # invert first without post-seismic
        #   Ain = np.delete(A,indexpo,1)
        #   mtemp = lst.lstsq(Ain,b,rcond=cond)[0]
    
        #   # rebuild full vector
        #   for z in range(len(indexpo)):
        #     mtemp = np.insert(mtemp,indexpo[z],0)
        #   minit = np.copy(mtemp)
        #   # # initialize bounds
        #   mmin,mmax = -np.ones(len(minit))*np.inf, np.ones(len(minit))*np.inf

        #   # We here define bounds for postseismic to be the same sign than steps
        #   # and steps inferior or egual to the coseimic initial
        #   for i in range(len(indexco)):
        #     if (pos[i] > 0.) and (minit[int(indexco[i])]>0.):
        #         mmin[int(indexpofull[i])], mmax[int(indexpofull[i])] = 0, np.inf
        #         mmin[int(indexco[i])], mmax[int(indexco[i])] = 0, minit[int(indexco[i])]
        #     if (pos[i] > 0.) and (minit[int(indexco[i])]<0.):
        #         mmin[int(indexpofull[i])], mmax[int(indexpofull[i])] = -np.inf , 0
        #         mmin[int(indexco[i])], mmax[int(indexco[i])] = minit[int(indexco[i])], 0

        # else:
        minit = lst.lstsq(A,b,rcond=None)[0]
        mmin,mmax = -np.ones(len(minit))*np.inf, np.ones(len(minit))*np.inf

        bounds=list(zip(mmin,mmax))
        # def eq_cond(x, *args):
        #    return (x[indexseast+1]/x[indexseast]) - (x[indexseas+1]/x[indexseas])

        ####Objective function and derivative
        _func = lambda x: np.sum(((np.dot(A,x)-b)/sigmad)**2)
        _fprime = lambda x: 2*np.dot(A.T/sigmad, (np.dot(A,x)-b)/sigmad)
        # if eguality:
        #     res = opt.fmin_slsqp(_func,minit,bounds=bounds,fprime=_fprime,eqcons=[eq_cond], \
        #         iter=iter,full_output=True,iprint=0,acc=acc)
        # else:
        res = opt.fmin_slsqp(_func,minit,bounds=bounds,fprime=_fprime, \
            iter=iter,full_output=True,iprint=0,acc=acc)
        fsoln = res[0]
        #print('Optimization:', fsoln)

    try:
       varx = np.linalg.inv(np.dot(A.T,A))
       res2 = np.sum(pow((b-np.dot(A,fsoln)),2))
       scale = 1./(A.shape[0]-A.shape[1])
       sigmam = np.sqrt(scale*res2*np.diag(varx))
    except:
       sigmam = np.ones((A.shape[1]))*float('NaN')
    return fsoln,sigmam


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    cube = arguments.get("--cube", "depl_cumule")
    lectfile = arguments.get("--lectfile", "lect.in")
    list_images = arguments.get("--list-image", "images_retenues")

    dates, dates_dec, bperp = read_list_images(list_images)
    base_dates = dates_dec - dates_dec[0]
    N = len(base_dates)

    ensure_gdal_header(cube, lectfile, N)
    last_band = open_gdal(cube, band=-1)
    first_date_dec = datetime.strptime(str(dates[0]), "%Y%m%d")

    # rms = arguments.get("--rms", None)
    # if rms is not None:
    #     rms = np.loadtxt(rms, comments='#', usecols=(2), dtype=float)
    
    print("Constructing base Time Model")
    ##> Define the model that will be used for all pixels
    # steps = None if arguments.get("--steps", None) is None else [(datetime.strptime(s, "%Y%m%d") - first_date_dec).days / 365.25 for s in arguments["--steps"].split(',')]
    # postseismic = None if arguments.get("--postseismic", None) is None else [float(s) for s in arguments["--postseismic"].split(',')]
    # slowslip = None if arguments.get("--slowslip", None) is None else [float(s) for s in arguments["--slowslip"].split(',')]
    # degreeday = None if arguments.get("--degreeday", None) is None else [float(s) for s in arguments["--degreeday"].split(',', 2)]
    # vector_file = [] if arguments.get('--vector', None) is None else arguments["--vector"].split(',')
    # vector = [np.loadtxt(f, comments='#', usecols=(1), dtype=float) for f in vector_file]
    # if len(vector) == 0:
    #     vector = None
    # bperp_vector = None if not arguments["--bperp"] else bperp

    # time_model = TimeModel.from_params(
    #     base_date=base_dates, 
    #     linear=arguments["--linear"], 
    #     seasonal=arguments["--seasonal"],
    #     seasonalt=arguments["--seasonal_increase"],
    #     semiannual=arguments["--semiannual"], 
    #     biannual=arguments["--biannual"], 
    #     steps=steps,
    #     postseismic=postseismic,
    #     slowslip=slowslip,
    #     vector=vector,
    #     bperp=bperp_vector,
    #     degreeday=degreeday)
    ##<
    
    print("Retrieving pixels Time Series")
    ##> Retrieve the pixels time series of interest and reference
    pixels_date_data = pixel_selection(cube, lectfile, N, arguments)
    print("Referencing Time Series")
    if arguments.get("<iref>", None) is not None and arguments.get("<jref>", None) is not None:
        pixels_date_data = ts_referencing(pixels_date_data, N, arguments)
    ##<

    mpl_dates = [datetime.strptime(str(d), "%Y%m%d") for d in dates]
    medians = [p[0] for p in pixels_date_data]
    q10 = [p[1] for p in pixels_date_data]
    q25 = [p[2] for p in pixels_date_data]
    q75 = [p[3] for p in pixels_date_data]
    q90 = [p[4] for p in pixels_date_data]
    for p in range(len(pixels_date_data)):
        plt.plot(mpl_dates, medians[p], '-k')
        plt.fill_between(mpl_dates, q25[p], q75[p], color="teal", alpha=0.4)
        plt.plot(mpl_dates, q10[p], '--r')
        plt.plot(mpl_dates, q90[p], '--r')
        plt.show()
