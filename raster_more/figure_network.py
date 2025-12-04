#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_network.py
------------

Usage: figure_network.py <aspsar> [--table=<table> --all-pairs=<all-pairs>]

Options:
-h | --help         Show this screen
"""

import docopt
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import dates as mdates
from matplotlib.dates import date2num
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

gdal.UseExceptions()


class Pair:
    def __init__(self, date1, date2, bp=None, b0=None):
        date1 = datetime.strptime(date1, "%Y%m%d")
        date2 = datetime.strptime(date2, "%Y%m%d")
        if date1 > date2:
            self.date1 = date2
            self.date2 = date1
        else:
            self.date1 = date1
            self.date2 = date2
        self.bperp = bp
        self.bperp0 = b0
        self.bt = None
        self._compute_bt()

    def _compute_bt(self):
        self.bt = (self.date2 - self.date1).days
        return self.bt
    
    def __eq__(self, value):
        return self.date1 == value.date1 and self.date2 == value.date2
    
    def __lt__(self, other):
        if self.date1 == other.date1:
            return self.date2 <= other.date2
        return self.date1 <= other.date1


def open_table(file):
    content = np.genfromtxt(file, skip_header=1, usecols=(0, 1), dtype=str)
    return content


def open_all_pairs(file):
    content = np.genfromtxt(file, skip_header=7, usecols=(0, 1, 6, 7, 8), dtype=str)
    return content


def plot_network(pairs):
    fig = plt.figure(figsize=(13,6))

    for p in pairs:
        dates = [p.date1, p.date2]
        bperp = [p.bperp0, p.bperp0 + p.bperp]
        plt.plot(dates, bperp, "-o", color='k')

    fig.suptitle("Interferogram network")
    fig.autofmt_xdate()
    plt.xlabel("Acquisition Date")
    plt.ylabel("Perpendicular Baseline (m)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))


def plot_histo(pairs):
    bt = [p.bt for p in pairs]
    plt.figure()
    plt.hist(bt, bins=100, color='teal')
    plt.ylabel("Baseline Occurence")
    plt.xlabel("Temporal Baseline (day)")


def savefig(path):
    plt.tight_layout()
    print("Saving:", path)
    plt.savefig(path, dpi=300)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    folder = arguments["<aspsar>"]
    table = arguments["--table"]
    if table is None:
        table = os.path.join(folder, 'PAIRS', 'table_pairs.txt')
    all_pairs_file = arguments["--all-pairs"]
    if all_pairs_file is None:
        all_pairs_file = os.path.join(folder, 'PAIRS', 'allPairsListing.txt')

    pairs = open_table(table)
    all_pairs = open_all_pairs(all_pairs_file)

    subset_pairs = [Pair(p[0], p[1]) for p in pairs]
    pairs = [Pair(p[0], p[1], b0=p[2], bp=p[3]) for p in all_pairs]
    # By sorting we align the sets
    subset_pairs.sort()
    pairs.sort()

    search_id = 0
    for p in subset_pairs:
        for ap in range(search_id, len(pairs)):
            current_pair = pairs[ap]
            if current_pair == p:
                p.bperp = float(current_pair.bperp)
                p.bperp0 = float(current_pair.bperp0)
                break

    plot_network(subset_pairs)

    plot_histo(subset_pairs)
    plt.show()
    
