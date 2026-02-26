#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correct_network_corr.py
------------

Usage: correct_network_corr.py <table> [<baselines>] --minbt=<minbt> --connectivity=<connectivity> --targets-bt=<targets-bt>

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


def open_baselines(file):
    ''' Open all pairs build by AMSTer
    '''
    data = np.genfromtxt(file, skip_header=7, dtype=str)
    baselines = []
    for d in data:
        print(d)
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
    baselines = arguments["<baselines>"]
    if baselines is None:
        baselines = "allPairsListing.txt"
    minbt = float(arguments["--minbt"])
    dconnectivity = arguments["--connectivity"]
    targets_bt = arguments["--targets-bt"]

    print("Fetching pairs: {}".format(baselines))
    pairs = open_baselines(baselines)
    pdates = fetch_all_dates(pairs)
    
    quit()

    print("Writing {} pairs: {}".format(len(selected_pairs), table))
    write_table(table, selected_pairs)
