#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################
# Author        : Leo L
############################################


"""\
plot_dates.py
-------------


Usage: plot_dates.py <datefile> --list_dates=<list_dates>
plot_dates.py  -h | --help

Options:
-h --help           Show this screen.
"""

# docopt (command line parser)
import docopt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as pltdates
from matplotlib.dates import date2num, num2date
from datetime import datetime

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    file = arguments["<datefile>"]
    dates_file = arguments["--list_dates"]
    dates_dec, rms = np.loadtxt(file, usecols=(0,1), dtype=float, comments='#', unpack=True)
    dates_str = np.loadtxt(dates_file, usecols=(0), dtype=str, comments='#')
    bp = np.loadtxt(dates_file, usecols=(3), dtype=float, comments='#')

    dates_dt = [datetime.strptime(d, '%Y%m%d') for d in dates_str]
    dates_ymod = [datetime.strptime(d[4:], '%m%d') for d in dates_str]

    rms_months = [[] for _ in range(12)]
    for k in range(len(dates_dt)):
        rms_months[dates_dt[k].month - 1].append(rms[k])
    rms_month_mean = [np.mean(l) for l in rms_months]
    print(rms_month_mean)
    # rms_month_std = [np.std(l) for l in rms_months]

    it_month = [0 for _ in range(12)]
    for k in range(len(dates_dt)):
        it_month[dates_dt[k].month - 1] += 1

    months_num = [date2num(datetime.strptime(d, '%m')) for d in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']]

    plt.figure()
    plt.plot(dates_dt, rms, '-ko')
    plt.title("RMS by dates")
    plt.xlabel("Dates")
    plt.ylabel("RMS")
    plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter("%Y/%m/%d"))
    plt.figure()
    plt.plot(dates_dt, bp)
    plt.title("Bperp by dates")
    plt.xlabel("Dates")
    plt.ylabel("Bperp")
    plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter("%Y/%m/%d"))
    plt.figure()
    plt.scatter(rms, bp)
    plt.title("rms vs bp")
    plt.xlabel("RMS")
    plt.ylabel("Bperp")
    plt.figure()
    plt.boxplot(rms_months, widths=8, positions=months_num, patch_artist=True, zorder=1)
    plt.title("RMS by yearly period (color by bperp)")
    plt.scatter(dates_ymod, rms, c=bp, cmap='magma', zorder=2, alpha=0.9, edgecolors="white", linewidths=0.6    )
    print("RMS mean +- 2sigma: {}+-{} (higher bound +2s {})".format(
        np.nanmean(rms),
        2*np.nanstd(rms),
        np.nanmean(rms) + 2 * np.nanstd(rms)
    ))
    plt.colorbar()
    plt.xlabel("Yearly month/period")
    plt.ylabel("RMS")
    plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter("%m"))
    plt.figure()
    plt.plot(months_num, it_month)
    plt.title("Occurence of images per months")
    plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter("%m"))
    plt.show()

