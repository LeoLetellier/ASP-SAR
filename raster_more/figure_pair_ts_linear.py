#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_pair_ts_linear.py
------------

Usage: figure_network.py --pairs=<p> --pairs-dir=<pd> --ts=<depl_cumule> --linear=<slope_cor_var> [--ref-zone=<r>] [--pref=<p> --suff=<s>] [--ndv=<ndv>] [--expected-range=<r>]

Options:
-h | --help         Show this screen
"""

import docopt
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from glob import glob
import os
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from datetime import datetime
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde


def open_gdal(file, band=1, supp_ndv=None, crop=None):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()

    if crop is None:
        data = band.ReadAsArray()
    else:
        data = band.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])
    data = data.astype(np.float32)
    
    cond = ~np.isfinite(data)
    if ndv is not None and ndv != np.nan:
        cond = cond & (data==ndv)
    if supp_ndv is not None and supp_ndv != np.nan:
        cond = cond & (data==supp_ndv)
    data = np.where(cond, np.nan, data)

    return data


def open_gdal_cube(file, supp_ndv=None, crop=None, ex_range=None):
    ds = gdal.Open(file)
    ndv = ds.GetRasterBand(1).GetNoDataValue()

    if crop is None:
        data = ds.ReadAsArray()
    else:
        data = ds.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])

    ds = None
    data = data.astype(np.float32)

    cond = ~np.isfinite(data)
    if ndv is not None and not np.isnan(ndv):
        cond = cond | (data == ndv)
    print(supp_ndv)
    if supp_ndv is not None and not np.isnan(supp_ndv):
        cond = cond | (data == supp_ndv)
    data = np.where(cond, np.nan, data)

    if ex_range is not None:
        data = np.where((data < ex_range[0]) | (data > ex_range[1]), np.nan, data)

    return data


def open_gdal_2bands(file, band1=1, band2=2, supp_ndv=None, crop=None):
    ds = gdal.Open(file)
    band1 = ds.GetRasterBand(band1)
    band2 = ds.GetRasterBand(band2)
    ndv1 = band1.GetNoDataValue()
    ndv2 = band2.GetNoDataValue()
    data = []

    for (b, n) in zip([band1, band2], [ndv1, ndv2]):
        if crop is None:
            d = b.ReadAsArray()
        else:
            d = b.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])
        d = d.astype(np.float32)
        
        cond = ~np.isfinite(d)
        if n is not None and n != np.nan:
            cond = cond | (d==n)
        if supp_ndv is not None and supp_ndv != np.nan:
            cond = cond | (d==supp_ndv)
        d = np.where(cond, np.nan, d)
        data.append(d)

    return data[0], data[1]


def std_gdal(file, band=1, supp_ndv=None, crop=None, ex_range=None):
    data = open_gdal(file, band, supp_ndv, crop)
    if ex_range is not None:
        data = np.where((data < ex_range[0]) | (data > ex_range[1]), np.nan, data)
    return np.nanstd(data)


def std_diff_gdal(file, band1, band2, supp_ndv=None, crop=None):
    data1, data2 = open_gdal_2bands(file, band1, band2, supp_ndv, crop)
    return np.nanstd(data2 - data1)


def get_date_list_from_ts(ts_file, as_dict=False):
    dataset = gdal.Open(ts_file)
    metadata = dataset.GetMetadata()
    result = {}

    for i in range(1, dataset.RasterCount + 1):
        date_key = f"Band_{i}"
        if date_key in metadata:
            if as_dict:
                result[metadata[date_key]] = i
            else:
                result[i - 1] = metadata[date_key]

    dataset = None

    if as_dict:
        return result
    else:
        # Convert the temporary dict to a list in band order
        return [result[i] for i in sorted(result.keys())]


def find_pairs(pairs_dir, pairs_list, pref, suff):
    if pref is None:
        pref = ""
    if suff is None:
        suff = ""
    print(pairs_list)
    pairs_infered = [os.path.join(pairs_dir, pref + p[0] + "_" + p[1] + suff) for p in pairs_list]
    pairs_list_updated = []
    pairs = []
    for (p, pl) in zip(pairs_infered, pairs_list):
        if not os.path.exists(p):
            print(f"File not found {p}")
        else:
            pairs.append(p)
            pairs_list_updated.append(pl)

    return pairs, pairs_list_updated


def check_dates(pairs, ts_file):
    dates = list(set(pairs.flatten()))
    dates_ts = get_date_list_from_ts(ts_file)
    dates.sort()
    dates_ts.sort()
    if dates != dates_ts:
        raise ValueError


def compute_std_single_pairs(pairs, pairs_resolved, crop, supp_ndv, ex_range=None):
    stds = []
    for p in tqdm(pairs_resolved):
        stds.append(std_gdal(p, crop=crop, supp_ndv=supp_ndv, ex_range=ex_range))
    return stds


def compute_std_simulated_pairs(pairs, ts_file, ts_cube):
    dates_ts = get_date_list_from_ts(ts_file, as_dict=True)
    bands = [[dates_ts[p[0]], dates_ts[p[1]]] for p in pairs]
    stds = []
    for b in tqdm(bands):
        stds.append(np.nanstd(ts_cube[b[1] - 1] - ts_cube[b[0] - 1]))
        if stds[-1] > 10:
            print(b)
    return stds


def figure_pair_ts_linear(pairs, pairs_dir, ts_file, linear_file, ref_zone, pref, suff, supp_ndv, ex_range):
    pairs = np.loadtxt(pairs, usecols=(0, 1), dtype=str)
    pairs = np.unique(pairs, axis=0)
    check_dates(pairs, ts_file)
    print("Load Cube")
    cube = open_gdal_cube(ts_file, crop=ref_zone, supp_ndv=supp_ndv, ex_range=ex_range)

    pairs_resolved, new_list = find_pairs(pairs_dir, pairs, pref, suff)
    print(pairs_resolved)

    std_single = compute_std_single_pairs(new_list, pairs_resolved, crop=ref_zone, supp_ndv=supp_ndv, ex_range=ex_range)
    std_simulated = compute_std_simulated_pairs(new_list, ts_file, cube)
    del cube
    bt = [(datetime.strptime(p[1], "%Y%m%d") - datetime.strptime(p[0], "%Y%m%d")).days for p in new_list]
    std_linear = std_gdal(linear_file, crop=ref_zone, supp_ndv=supp_ndv, ex_range=ex_range)

    bt = np.asarray(bt) / 365.25
    std_single = np.asarray(std_single) / bt
    std_simulated = np.asarray(std_simulated) / bt
    # std_single[std_single>10]=np.nan
    # std_simulated[std_simulated>10]=np.nan

    figure_stds(std_single, std_simulated, std_linear, bt)
    plt.show()


def figure_stds(std_pairs, std_pairs_ts, std_linear, bt, bins=50, cmap='viridis',
                 as_pdf=True, pdf_points=200):
    std_pairs = np.asarray(std_pairs)
    std_pairs_ts = np.asarray(std_pairs_ts)
    bt = np.asarray(bt)

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(
        4, 4,
        hspace=0.05, wspace=0.05,
        width_ratios=[1, 1, 1, 0.25],
        height_ratios=[0.25, 1, 1, 1],
    )

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[0, 3])

    mask = np.isfinite(std_pairs) & np.isfinite(std_pairs_ts) & np.isfinite(bt)
    if not mask.all():
        print(f"Dropping {(~mask).sum()} non-finite entries")
        std_pairs = std_pairs[mask]
        std_pairs_ts = std_pairs_ts[mask]
        bt = bt[mask]

    pos_mask = bt > 0
    if not pos_mask.all():
        print(f"Dropping {(~pos_mask).sum()} non-positive bt entries (invalid for log color scale)")
        std_pairs = std_pairs[pos_mask]
        std_pairs_ts = std_pairs_ts[pos_mask]
        bt = bt[pos_mask]

    norm = LogNorm(vmin=bt.min(), vmax=bt.max())
    sc = ax_main.scatter(std_pairs, std_pairs_ts, s=20, alpha=0.8, c=bt, cmap=cmap, norm=norm, label='data')

    lo = 0
    hi = max(std_pairs.max(), std_pairs_ts.max(), std_linear) * 1.05
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)

    ax_main.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='1:1 line')

    ax_main.axvline(std_linear, color='red', linestyle=':', linewidth=1)
    ax_main.axhline(std_linear, color='red', linestyle=':', linewidth=1)
    ax_main.scatter([std_linear], [std_linear], s=100, facecolors='none',
                     edgecolors='red', linewidths=2, zorder=5, label=f'linear={std_linear:.2g}m/y')

    ax_main.set_xlabel('Single Pair Standard Deviation [m/y]')
    ax_main.set_ylabel('Simulated Pair Standard Deviation [m/y]')
    ax_main.legend(loc='upper left', fontsize=8)

    cbar = fig.colorbar(sc, cax=ax_cbar)
    cbar.set_label('bt', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # --- marginal distributions: smooth PDF via KDE, or fall back to hist bars ---
    grid = np.linspace(lo, hi, pdf_points)

    if as_pdf:
        kde_x = gaussian_kde(std_pairs)
        ax_histx.fill_between(grid, kde_x(grid), color='steelblue', alpha=0.4)
        ax_histx.plot(grid, kde_x(grid), color='steelblue', linewidth=1.5)

        kde_y = gaussian_kde(std_pairs_ts)
        ax_histy.fill_betweenx(grid, kde_y(grid), color='steelblue', alpha=0.4)
        ax_histy.plot(kde_y(grid), grid, color='steelblue', linewidth=1.5)
    else:
        ax_histx.hist(std_pairs, bins=bins, range=(lo, hi), color='steelblue', alpha=0.7, density=True)
        ax_histy.hist(std_pairs_ts, bins=bins, range=(lo, hi), orientation='horizontal',
                      color='steelblue', alpha=0.7, density=True)

    ax_histx.axvline(std_linear, color='red', linestyle=':', linewidth=1)
    ax_histx.tick_params(axis='x', labelbottom=False)

    ax_histy.axhline(std_linear, color='red', linestyle=':', linewidth=1)
    ax_histy.tick_params(axis='y', labelleft=False)

    return fig


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    pairs = arguments["--pairs"]
    pairs_dir = arguments["--pairs-dir"]
    ts_file = arguments["--ts"]
    linear_file = arguments["--linear"]
    ref_zone = arguments["--ref-zone"]
    if ref_zone is not None:
        ref_zone = [int(k) for k in ref_zone.split(',')]
    pref = arguments["--pref"]
    suff = arguments["--suff"]
    ndv = float(arguments["--ndv"]) if arguments["--ndv"] is not None else None
    ex_range = arguments["--expected-range"]
    if ex_range is not None:
        ex_range = [float(k) for k in ex_range.split(",")]

    figure_pair_ts_linear(pairs, pairs_dir, ts_file, linear_file, ref_zone, pref, suff, ndv, ex_range)
