#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_amplis.py
Usage: check_amplis.py <pattern> [--crop=<c>]
check_amplis.py -h | --help
Options:
-h --help             Show this screen
"""
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import docopt
from glob import glob
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
gdal.UseExceptions()


def open_gdal(file, band=1, crop=None):
    ds = gdal.Open(file)
    bd = ds.GetRasterBand(band)
    if crop is None:
        array = bd.ReadAsArray()
    else:
        array = bd.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])
    return array


def half_month_key(d):
    """Return (month, half) where half=1 for days 1-15, half=2 for days 16-end."""
    return (d.month, 1 if d.day <= 15 else 2)


def half_month_x(month, half):
    """Dummy-year x position for a half-month bin, for plotting on a Jan-Dec axis."""
    day = 1 if half == 1 else 16
    return datetime(2000, month, day)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    pattern = arguments["<pattern>"]
    crop = arguments["--crop"]
    if crop is not None:
        crop = [int(k) for k in crop.split(',')]

    files = sorted(glob(pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {pattern}")

    date = [f.split('_')[2] for f in files]
    date_dt = [datetime.strptime(d, "%Y%m%d") for d in date]

    mean = np.zeros(shape=len(files))
    median = np.zeros(shape=len(files))
    std = np.zeros(shape=len(files))
    p10 = np.zeros(shape=len(files))
    p90 = np.zeros(shape=len(files))

    for i, f in tqdm(list(enumerate(files))):
        data = open_gdal(f, crop=crop)
        mean[i] = np.mean(data)
        median[i] = np.median(data)
        std[i] = np.std(data)
        p10[i], p90[i] = np.percentile(data, (10, 90))

    # --- Raw per-date plot ---
    fig, axs = plt.subplots(2, 1)
    axs[0].errorbar(date_dt, mean, yerr=std, label='mean', fmt='o',
                     linestyle='none', capsize=3, markersize=5)
    axs[1].errorbar(date_dt, median, yerr=(median - p10, p90 - median), label='median',
                     fmt='o', linestyle='none', capsize=3, markersize=5)
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()

    # --- Half-month aggregation across all years ---
    bins = defaultdict(lambda: {"mean": [], "median": [], "std": [], "p10": [], "p90": []})

    for i, d in enumerate(date_dt):
        key = half_month_key(d)
        bins[key]["mean"].append(mean[i])
        bins[key]["median"].append(median[i])
        bins[key]["std"].append(std[i])
        bins[key]["p10"].append(p10[i])
        bins[key]["p90"].append(p90[i])

    sorted_keys = sorted(bins.keys())  # sorts by (month, half) -> Jan H1, Jan H2, Feb H1, ...

    x = [half_month_x(m, h) for (m, h) in sorted_keys]
    n_obs = [len(bins[k]["mean"]) for k in sorted_keys]

    mean_agg = np.array([np.mean(bins[k]["mean"]) for k in sorted_keys])
    mean_std = np.array([np.std(bins[k]["mean"]) for k in sorted_keys])  # spread across years, not within-image std

    median_agg = np.array([np.mean(bins[k]["median"]) for k in sorted_keys])
    p10_agg = np.array([np.mean(bins[k]["p10"]) for k in sorted_keys])
    p90_agg = np.array([np.mean(bins[k]["p90"]) for k in sorted_keys])

    fig2, axs2 = plt.subplots(2, 1)
    axs2[0].errorbar(x, mean_agg, yerr=mean_std, label='mean (half-month avg)',
                      fmt='o', linestyle='none', capsize=3, markersize=5)
    axs2[1].errorbar(x, median_agg, yerr=(median_agg - p10_agg, p90_agg - median_agg),
                      label='median (half-month avg)', fmt='o', linestyle='none', capsize=3, markersize=5)

    for ax in axs2:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.legend()

    # annotate number of observations per bin on the mean subplot
    for xi, yi, n in zip(x, mean_agg, n_obs):
        axs2[0].annotate(str(n), (xi, yi), textcoords="offset points", xytext=(0, 8), fontsize=8, ha='center')

    fig2.autofmt_xdate()
    plt.tight_layout()
    plt.show()