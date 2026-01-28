#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_connectivity.py
------------

Usage: network_connectivity.py <table> [--save-dir=<save-dir>]

Options:
-h | --help         Show this screen
"""

import docopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from cmcrameri import cm
import os


def read_pairs_file(file):
    try:
        # Try to load d1 / d2 / bp / bt
        try:
            pairs = np.loadtxt(file, skiprows=2, usecols=(0, 1, 2, 3), delimiter='\t', dtype=str)
        except:
            pairs = np.loadtxt(file, skiprows=2, usecols=(0, 1, 2, 3), delimiter=' ', dtype=str)
    except:
        # Else load d1 / d2
        try:
            pairs = np.loadtxt(file, skiprows=2, usecols=(0, 1), delimiter='\t', dtype=str)
        except:
            pairs = np.loadtxt(file, skiprows=2, usecols=(0, 1), delimiter=' ', dtype=str)
    return pairs


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    table = arguments["<table>"]
    save = arguments["--save-dir"]

    pairs = read_pairs_file(table)
    print("Number of pairs", len(pairs))
    dates1 = pairs[:, 0]
    dates2 = pairs[:, 1]
    all_dates = list(set(list(dates1) + list(dates2)))
    print("Number of dates", len(all_dates))
    all_dates.sort()

    # Connections per date
    connection_date = np.zeros(len(all_dates))
    for d in range(len(all_dates)):
        count = 0
        for k in dates1:
            if k == all_dates[d]:
                count += 1
        for k in dates2:
            if k == all_dates[d]:
                count += 1
        connection_date[d] = count

    dates1 = [datetime.strptime(d, "%Y%m%d") for d in dates1]
    dates2 = [datetime.strptime(d, "%Y%m%d") for d in dates2]
    bt = [(d2 - d1).days for (d2, d1) in zip(dates2, dates1)]
    all_dates = [datetime.strptime(d, "%Y%m%d") for d in all_dates]

    step_count = np.zeros((13, len(all_dates) - 1))
    for d in range(len(all_dates) - 1):
        step1 = all_dates[d]
        step2 = all_dates[d + 1]
        for p in range(len(pairs)):
            if dates1[p] < dates2[p]:
                d1 = dates1[p]
                d2 = dates2[p]
            else:
                d1 = dates2[p]
                d2 = dates1[p]
            if not (d2 < step1 or d1 > step2):
                baseline = (dates2[p] - dates1[p]).days
                if baseline < 1 * 30.5:
                    step_count[0, d] += 1
                elif baseline < 2 * 30.5:
                    step_count[1, d] += 1
                elif baseline < 3 * 30.5:
                    step_count[2, d] += 1
                elif baseline < 4 * 30.5:
                    step_count[3, d] += 1
                elif baseline < 5 * 30.5:
                    step_count[4, d] += 1
                elif baseline < 6 * 30.5:
                    step_count[5, d] += 1
                elif baseline < 7 * 30.5:
                    step_count[6, d] += 1
                elif baseline < 8 * 30.5:
                    step_count[7, d] += 1
                elif baseline < 9 * 30.5:
                    step_count[8, d] += 1
                elif baseline < 10 * 30.5:
                    step_count[9, d] += 1
                elif baseline < 11 * 30.5:
                    step_count[10, d] += 1
                elif baseline < 12 * 30.5:
                    step_count[11, d] += 1
                else:
                    step_count[12, d] += 1

    all_dates = np.array(all_dates)
    inter = all_dates[:-1] + (all_dates[:-1] - all_dates[1:]) / 2
    inter = [(start + (end - start) / 2) for start, end in zip(all_dates[:-1], all_dates[1:])]
    cumtot = np.sum(step_count, axis=0)
    cmap = cm.navia
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # colors = [cmap(k) for k in bounds]
    colors = [cmap(k / 12) for k in range(13)]
    bottom = np.zeros(len(all_dates) - 1)
    fig, ax = plt.subplots(figsize=(12,8))
    for k in range(13):
        cum = np.sum(step_count[:k + 1], axis=0)
        ax.fill_between(inter, cum, bottom, color=colors[k])
        bottom = cum
    ax.plot(inter, cumtot, color='k')
    ax.plot(all_dates, 3 * np.ones(shape=len(all_dates)), color='firebrick', alpha=0.7, ls='--')
    ax.plot(all_dates, connection_date, color='lightcoral', alpha=0.8, lw=3)
    ax.set_title("Pair Coverage")
    ax.set_xlabel("Date Interval Centered")
    ax.set_ylabel("Pair Count (Stack) / Date Connectivity (Curve)")
    
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="max")
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=c, orientation='vertical',
             label="Temporal Baselines (month)",
             fraction=0.02, pad=0.04)

    plt.tight_layout()
    if save is not None:
        plt.savefig(os.path.join(save, os.path.splitext(table) + "_connectivity.pdf", dpi=300))
    plt.show()
