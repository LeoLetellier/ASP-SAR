#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster_generate_network.py
------------

Usage: amster_generate_network.py <table> [<baselines>] --maxbp=<maxbp> --maxbt=<maxbt> --minbt=<minbt> [--epsilon=<epsilon>] [--maxperdate=<maxperdate>] [--add-connectivity] [--add-3-6-months] [--restrain-nb]

Options:
-h | --help         Show this screen
<table>             table*
<baselines>         all pair listing
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


def nb_allowed_bt(length, total_duration):
    return 6 / length * total_duration


def central_date(date1, date2):
    d1 = datetime.strptime(str(date1), "%Y%m%d")
    d2 = datetime.strptime(str(date2), "%Y%m%d")
    center = d1 + (d2 - d1) / 2
    return int(center.strftime("%Y%m%d"))


def reduce_to_allowed_number(pairs, length, total_duration):
    central_time = [central_date(p.date1, p.date2) for p in pairs]
    allowed_number = nb_allowed_bt(length, total_duration)
    initial_nb = len(pairs)
    if len(pairs) <= 2 * allowed_number:
        return pairs
    # between one time and two times the allowed number
    decimation_factor = int(len(pairs) // allowed_number)
    indexed_times = [(time, idx) for idx, time in enumerate(central_time)]
    sorted_times = sorted(indexed_times, key=lambda x: x[0])
    decimate_index = [s[1] for s in sorted_times][::decimation_factor]
    pairs = [pairs[i] for i in decimate_index]
    print("Decimated from {} inital pairs to {} pairs".format(initial_nb, len(pairs)))
    return pairs


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    print("Generating long baselines network for AMSTer")

    table = arguments["<table>"]
    baselines = arguments["<baselines>"]
    if baselines is None:
        baselines = "allPairsListing.txt"
    maxbp = float(arguments["--maxbp"])
    maxbt = float(arguments["--maxbt"])
    minbt = float(arguments["--minbt"])
    epsilon = arguments["--epsilon"]
    if epsilon is None:
        epsilon = 0.02
    else:
        epsilon = float(epsilon)
    maxperdate = arguments["--maxperdate"]
    if maxperdate is not None:
        maxperdate = int(maxperdate)
    do_connectivity = arguments["--add-connectivity"]
    do_3_6 = arguments["--add-3-6-months"]
    restrain_nb = arguments["--restrain-nb"]

    print("Fetching pairs: {}".format(baselines))
    pairs = open_baselines(baselines)
    pdates = fetch_all_dates(pairs)
    print("Filtering pairs ({})".format(len(pairs)))
    duration_year = (datetime.strptime(max(pdates), "%Y%m%d") - datetime.strptime(min(pdates), "%Y%m%d")).days / 365.25
    # Filter out of range bt and bp
    pairs = [p for p in pairs if (abs(p.bp) <= maxbp and abs(p.bt) >= minbt and abs(p.bt) <= maxbt)]

    print("Selecting pairs ({})".format(len(pairs)))
    selected_pairs = []

    # multi-yearly baselines
    max_year = min(int(duration_year) + 1, int(maxbt // 365.25))
    for k in range(max_year, 1, -1):
        s = select_bt(pairs, k * 365.25, epsilon=epsilon * 0.005 + 0.001 / np.square(k))
        s = filter_pselection_per_bp(s, pdates, maxperdate)
        if restrain_nb:
            s = reduce_to_allowed_number(s, k, duration_year)
        selected_pairs += s
        print("{}Y: {}".format(k, len(s)))

    # selected_pairs = filter_pselection_per_bp(selected_pairs, pdates, maxperdate, square=True)
    # 1 year baselines
    s = select_bt(pairs, k * 365.25, epsilon=epsilon * 0.90)
    s = filter_pselection_per_bp(s, pdates, maxperdate)
    if restrain_nb:
        s = reduce_to_allowed_number(s, k, duration_year)
    selected_pairs += s
    print("1Y: {}".format(len(s)))

    if do_3_6:
        # 6 months baselines
        s = select_bt(pairs, 182.625, epsilon=epsilon)
        s = filter_pselection_per_bp(s, pdates, maxperdate)
        selected_pairs += s
        print("6m: {}".format(len(s)))
        
        # 3 months baselines
        s = select_bt(pairs, 91.3125, epsilon=epsilon)
        limit = None
        if maxperdate is not None:
            limit = maxperdate + 1
        s = filter_pselection_per_bp(s, pdates, limit)
        selected_pairs += s
        print("3m: {}".format(len(s)))

    if do_connectivity:
        # 1 month baselines
        s = select_bt(pairs, 60, epsilon=0.5, gmax=30)
        seen_dates = dict.fromkeys(pdates, 0)
        add_connectivity = []
        for p in selected_pairs:
            seen_dates[p.date1] += 1
            seen_dates[p.date2] += 1
        for (d, n) in seen_dates.items():
            if n < 6:
                missing_nb = (6 - n) // 2
                add_connectivity += sorted(select_pair_per_date(s, d), key=lambda x: 10 * x.bp * x.bp + x.bt * x.bt)[:missing_nb]
        print("Add connectivity links ({})".format(len(add_connectivity)))
        selected_pairs += add_connectivity

    before_reduction = len(selected_pairs)
    selected_pairs = list(set(selected_pairs))
    print("Removing redondant pairs (-{})".format(before_reduction - len(selected_pairs)))

    if maxperdate is not None:
        removing_pairs = []
        for d in pdates:
            p = select_pair_per_date(selected_pairs, d)
            if len(p) <= 2:
                removing_pairs += select_pair_per_date(selected_pairs, d)
        removing_pairs = list(set(removing_pairs))
        print("Reducing number of pairs per date (-{})".format(len(removing_pairs)))
        selected_pairs = [p for p in selected_pairs if p not in removing_pairs]

    print("Writing {} pairs: {}".format(len(selected_pairs), table))
    write_table(table, selected_pairs)
