#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_depl_cumule.py 

Usage: plot_depl_cumule.py <depl> [<dates>]
plot_depl_cumule.py -h | --help

Options:
-h | --help             Show this screen
"""

import docopt
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from matplotlib import dates as mdates
from matplotlib.dates import date2num
from datetime import datetime


def open_depl(path):
    ds = gdal.Open(path)
    datas = []

    for b in range(1, ds.RasterCount + 1):
        print(f"\tband {b}")
        band = ds.GetRasterBand(b)
        ndv = band.GetNoDataValue()
        data = band.ReadAsArray()
        if ndv is not None:
            data[data == ndv] = np.nan
        datas.append(data.flatten())
    
    return datas


def open_dates(path):
    with open(path, 'r') as infile:
        content = list(filter(None, infile.read().split('\n')))
    dates = [date2num(datetime.strptime(c.split(' ')[0], "%Y%m%d")) for c in content]
    return dates


def plot_depl(data, dates):
    fig = plt.figure()
    # for p in range(len(data)):
    #     date = dates[p]
    #     d = data[p]
        # plt.scatter(np.full(shape=d.shape, fill_value=date), d)

    plt.fill_between(dates, data[:, 0] + data[:, 1], data[:, 0] - data[:, 1], color='k', alpha=0.2)
    plt.scatter(dates, data[:, 2], color='blue')
    plt.scatter(dates, data[:, 3], color='red')
    plt.plot(dates, data[:, 0])

    fig.autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    plt.tight_layout()


def quick_clean(data):
    for d in data:
        d[d > 40] = np.nan
        d[d < -40] = np.nan
    return data


def summarize(data):
    ndata = []
    for d in data:
        mean = np.nanmean(d)
        std = np.nanstd(d)
        (p02, p98) = np.nanpercentile(data, (2, 98))
        ndata.append([mean, std, p02, p98])
    return np.array(ndata)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    depl = arguments["<depl>"]
    dates = arguments["<dates>"]
    
    print("Read cube...")
    data = open_depl(depl)
    print("Read dates...")
    dates = open_dates(dates)

    data = quick_clean(data)
    data = summarize(data)

    print("Plotting...")
    plot_depl(data, dates)
    plt.show()
