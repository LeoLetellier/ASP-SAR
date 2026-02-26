#!/usr/bin/env python3

"""
invert_ts_period.py
-------------

Usage: 
    invert_ts_period.py <cols> <lines> <period-cuts> [--cube=<cube> --list_images=<list_images> --windowsize=<windowsize> \
(<iref> <jref> --windowrefsize=<windowrefsize>) --rms=<rms> --lectfile=<lectfile>]


invert_ts_period.py -h | --help
Options:

-h --help               Show this screen
"""

print()
print("Leo Letellier (2026)")
print()
print("Modified from:")
print('Simon Daout')
print("PyGdalSAR: invert_disp_pixel.py")
print()


import numpy as np
import scipy.optimize as opt
import numpy.linalg as lst
import os
import docopt
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import bisect
gdal.UseExceptions()


def invert_linear(start_time, time, values, sigma=None, reference=None):
    valid_values = np.where(np.isnan(values), False, True)
    values = values[valid_values]
    time = time[valid_values]
    ref_time = time - start_time
    if sigma is None:
        sigma = np.ones(shape=values.shape)
    else:
        sigma = sigma[valid_values]
    
    linear = ref_time.copy()
    if reference is None:
        reference = np.ones(shape=values.shape)
        model = np.column_stack([reference, linear])
    else:
        values -= reference
        model = np.column_stack([linear])

    coeffs, sigma_out = invert_simple(model, values, sigma)
    return coeffs, sigma_out


def read_list_images(list_images):
    dates = np.loadtxt(list_images, comments='#', usecols=(1), unpack=True, dtype=str)
    dates_dec, bperp = np.loadtxt(list_images, comments='#', usecols=(3,5), unpack=True)
    return dates, dates_dec, bperp


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
            data_med = np.nanmedian(data, axis=1)
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


def invert_simple(A, B, sigma):
    """ Solve A.X = B using Sequential Least Squares & Error Propagation
    """
    minit = lst.lstsq(A,B,rcond=None)[0]
    mmin, mmax = -np.ones(len(minit)) * np.inf, np.ones(len(minit)) * np.inf
    bounds = list(zip(mmin, mmax))
    _func = lambda x: np.sum(((np.dot(A, x) - B) / sigma)**2)
    _fprime = lambda x: 2 * np.dot(A.T / sigma, (np.dot(A, x) - B) / sigma)
    res = opt.fmin_slsqp(_func, minit, bounds=bounds, fprime=_fprime, iter=100, full_output=True, iprint=0)
    coeffs = res[0]

    try:
       varx = np.linalg.inv(np.dot(A.T, A))
       res2 = np.sum(pow((B - np.dot(A,coeffs)), 2))
       scale = 1 / (A.shape[0] - A.shape[1])
       sigma_out = np.sqrt(scale * res2 * np.diag(varx))
    except:
       sigma_out = np.ones((A.shape[1])) * float('NaN')

    return coeffs, sigma_out


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    cube = arguments["--cube"]
    if cube is None:
        cube = "depl_cumule"
    lectfile = arguments["--lectfile"]
    if lectfile is None:
        lectfile = "lect.in"
    list_images = arguments["--list_images"]
    if list_images is None:
        list_images = "images_retenues"

    dates, dates_dec, bperp = read_list_images(list_images)
    dates_dt = [datetime.strptime(d, "%Y%m%d") for d in dates]
    base_dates = dates_dec - dates_dec[0]
    N = len(base_dates)

    ensure_gdal_header(cube, lectfile, N)
    first_date_dec = dates_dt[0]

    rms = arguments.get("--rms", None)
    if rms is not None:
        rms = np.loadtxt(rms, comments='#', usecols=(2), dtype=float)
    
    print("Constructing Velocity per Period Time Model")
    ##> Periods should be MMDD for month and day in the year
    working_years = list(set([str(d)[:4] for d in dates]))
    working_years.sort()
    print("Years:", working_years)
    period_cuts = arguments["<period-cuts>"].split(",")
    periods = [dates[0]]
    for y in working_years:
        periods += [y + p for p in period_cuts]
    periods += [dates[-1]]
    unit_periods = [[periods[k], periods[k + 1]] for k in range(len(periods) - 1)]
    print("Using {} unit periods".format(len(unit_periods)))
    ##<
    
    ##> Retrieve the pixels time series of interest and reference
    print("Retrieving pixels Time Series")
    pixels_date_data = pixel_selection(cube, lectfile, N, arguments)
    if arguments.get("<iref>", None) is not None and arguments.get("<jref>", None) is not None:
        print("Referencing Time Series")
        pixels_date_data = ts_referencing(pixels_date_data, N, arguments)
    ##<

    for i, point in enumerate(pixels_date_data):
        print("Point [{}]".format(i + 1))
        reference = None
        lins = []
        refs = []
        
        plt.figure(figsize=(14,8))
        for j, p in enumerate(unit_periods):
            print("\tPeriod {} to {}".format(p[0], p[1]))
            first_date_period, last_date_period = datetime.strptime(p[0], "%Y%m%d"), datetime.strptime(p[1], "%Y%m%d")
            r1 = bisect.bisect_left(dates_dt, first_date_period)
            r2 = bisect.bisect_right(dates_dt, last_date_period)
            times = dates_dec[r1:r2]
            values = point[r1:r2]
            coeffs, sigmas = invert_linear(times[0], times, values, reference=reference)
            if len(coeffs) > 1:
                reference = coeffs[0]
                lins.append(coeffs[1])
                refs.append(reference)
            else:
                lins.append(coeffs[0])
                refs.append(reference)
            reference += lins[-1] * (times[-1] - times[0])
            print(coeffs, sigmas)
            if not np.isnan(sigmas[0]):
                plt.plot([first_date_period, last_date_period], [refs[-1], refs[-1] + (times[-1] - times[0]) * lins[-1]], '-ko')
            plt.scatter(dates_dt[r1:r2], values, color='lightcoral')
        print(lins)
    plt.show()

