#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster_generate_network.py
------------

Usage: amster_generate_network.py <table> [--maxbp=<maxbp> --maxbt=<maxbt> --minbt=<minbt>] --outfile=<outfile> [--it=<it>]

Options:
-h | --help         Show this screen
<table>             table*
<baselines>         SM_Approx_baselines
"""

import docopt
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.dates import date2num
import numpy as np


class Pair:
    def __init__(self, date1: str, date2: str, bp: float, bt: float):
        self.date1 = date1
        self.date2 = date2
        self.bp = bp
        self.bt = bt

    def get_line(self) -> str:
        return "\n{}\t{}\t{}\t\t{}\n".format(self.date1, self.date2, self.bp, self.bt)
    
    def __eq__(self, value):
        return (self.date1, self.date2, self.bp, self.bt) == (value.date1, value.date2, value.bp, value.bt)
    
    def __hash__(self):
        return hash((self.date1, self.date2, self.bp, self.bt))


def write_table(file, selected_pairs: list[Pair]):
    ''' Save the resulting pair network to file (AMSTer table format)
    '''
    content = "Master\tSlave\tBperp\tDelay\n"
    for p in selected_pairs:
        content += p.get_line()
    with open(file, 'w') as outfile:
        outfile.write(content)


def open_table(file):
    data = np.genfromtxt(file, skip_header=1, usecols=(0, 1, 2, 3), dtype=str)
    pairs = []
    for d in data:
        pairs.append(Pair(d[0], d[1], float(d[2]), float(d[3])))
    return pairs


def open_baselines(file):
    ''' Open all pairs build by AMSTer
    '''
    data = np.genfromtxt(file, skip_header=7, dtype=str)
    baselines = []
    for d in data:
        pair = Pair(d[0], d[1], float(d[7]), float(d[8]))
        baselines.append(pair)
    return baselines


def select_bt(baselines: list[Pair], period: float, epsilon: float = 0.02, gmax=30, gmin=2):
    ''' Select baseline corresponding to a specific period, by a small margin
    '''
    selected = []
    if epsilon * period > gmax:
        epsilon = gmax / period
    elif epsilon * period < gmin:
        epsilon = gmin / period
    for b in baselines:
        if abs(b.bt - period) / period < epsilon:
            selected.append(b)
    return selected


def fetch_all_dates(pairs):
    ''' Find all dates involved into a pair list
    '''
    dates = []
    for p in pairs:
        dates.append(p.date1)
        dates.append(p.date2)
    return list(set(dates))


def select_pair_per_date(pairs, date, inverse=False):
    ''' Find all pairs involving a specific date
    '''
    s = []
    for p in pairs:
        if inverse:
            if p.date1 != date and p.date2 != date:
                s.append(p)
        else:
            if p.date1 == date or p.date2 == date:
                s.append(p)
    return s


def filter_pselection_per_bp(selection, dates, limit, square=False):
    ''' Filter a pair selection by selecting those per date with the smaller bp and/or bt
    '''
    if limit is not None:
        restricted = []
        for d in dates:
            p = select_pair_per_date(selection, d)
            if square:
                restricted += sorted(p, key=lambda x: (10 * x.bp * x.bp + x.bt * x.bt) / (x.bt * x.bt))[:limit]
            else:
                restricted += sorted(p, key=lambda x: abs(x.bp))[:limit]
        return restricted
    return selection


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    print("Generating long baselines network for AMSTer")

    table = arguments["<table>"]
    outfile = arguments["--outfile"]
    maxbp = arguments["--maxbp"]
    maxbt = arguments["--maxbt"]
    minbt = arguments["--minbt"]
    it = arguments["--it"]
    it = 2 if it is None else int(it)

    pairs = open_table(table)

    print("Loading {} pairs".format(len(pairs)))
    print("Removing pairs:")

    do_filter = maxbp is not None or minbt is not None or maxbt is not None
    if do_filter:
        previous_len = len(pairs)
        if maxbp is not None:
            pairs = [p for p in pairs if abs(p.bp) <= float(maxbp)]
        if maxbt is not None:
            pairs = [p for p in pairs if p.bt <= float(maxbt)]
        if maxbp is not None:
            pairs = [p for p in pairs if p.bt >= float(minbt)]

        print("\t{}\t: pairs exceeding bp and bt thresholds".format(previous_len - len(pairs)))

    before_reduction = len(pairs)
    selected_pairs = list(set(pairs))
    print("\t{}\t: redondant pairs".format(before_reduction - len(selected_pairs)))

    for i in range(it):
        removing_pairs = []
        pdates = fetch_all_dates(pairs)
        for d in pdates:
            p = select_pair_per_date(selected_pairs, d)
            if len(p) <= 2:
                removing_pairs += select_pair_per_date(selected_pairs, d)
        removing_pairs = list(set(removing_pairs))
        print("\t{}\t: pairs associated to dates with less than 3 pairs (it:{})".format(len(removing_pairs), i + 1))
        selected_pairs = [p for p in selected_pairs if p not in removing_pairs]

    print("Keeping {} pairs.".format(len(selected_pairs)))
    write_table(outfile, selected_pairs)
    print("Saving {}.".format(outfile))
