#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network2.py
------------
Backward and forward connection

Usage: network2.py <table> <baselines> --bt-target=<bt-target> --bt-min=<bt-min> --bt-max=<bt-max> [--existing-network=<existing-network>]

Options:
-h | --help         Show this screen
"""

import docopt
from datetime import datetime
import numpy as np


class Pair:
    def __init__(self, date1: str, date2: str, bp: float, bt: float):
        self.date1 = date1
        self.date2 = date2
        self.bt = bt
        self.bp = bp

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


def open_dates(file):
    dates = np.loadtxt(file, dtype=str, usecols=(0))
    return dates


def open_table(file):
    data = np.genfromtxt(file, skip_header=1, usecols=(0, 1, 2, 3), dtype=str)
    pairs = []
    for d in data:
        pairs.append(Pair(d[0], d[1], float(d[2]), float(d[3])))
    return pairs


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


class DPair:
    def __init__(self, date1, date2):
        self.date1 = date1
        self.date2 = date2
        self.bt = (datetime.strptime(date1, "%Y%m%d") - datetime.strptime(date2, "%Y%m%d")).days
    

def match_pair(date, dates, target, bt_min, bt_max, existing_connections=0):
    backward_pair, forward_pair, bonus = None, None, None
    if existing_connections < 2:
        bt = [DPair(d, date) for d in dates]
        bt = [b for b in bt if b.bt > bt_min and b.bt < bt_max]
        forward_bt = [b for b in bt if b.bt - target < 0]
        backward_bt = [b for b in bt if b.bt - target > 0]
        forward_bt.sort(key=lambda b: abs(b.bt - target))
        backward_bt.sort(key=lambda b: abs(b.bt - target))
        forward_pair = forward_bt[0] if len(forward_bt) > 0 else None
        backward_pair = backward_bt[0] if len(backward_bt) > 0 else None
        # bonus = forward_bt[1] if len(forward_bt) > 1 else None
    elif existing_connections == 2:
        bt = [DPair(d, date) for d in dates]
        bt = [b for b in bt if b.bt > bt_min and b.bt < bt_max]
        forward_pair = bt[0] if len(bt) > 0 else None
    return backward_pair, forward_pair, bonus


def find_pair_from_dates(d1, d2, pairs):
    for p in pairs:
        if d1 in [p.date1, p.date2]:
            if d2 in [p.date1, p.date2]:
                return p


def count_pairs_per_date(pairs, date):
    count = 0
    for p in pairs:
        if date in [p.date1, p.date2]:
            count += 1
    return count


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    table = arguments["<table>"]
    baselines = arguments["<baselines>"]
    if baselines is None:
        baselines = "allPairsListing.txt"
    target = float(arguments["--bt-target"])
    bt_min = float(arguments["--bt-min"])
    bt_max = float(arguments["--bt-max"])
    existing_network = arguments["--existing-network"]
    
    previous_pairs = None
    if existing_network is not None:
        previous_pairs = open_table(existing_network)

    bpairs = open_baselines(baselines)
    dates = fetch_all_dates(bpairs)
    pairs = []

    for d in dates:
        count = 0
        if previous_pairs is not None:
            count = count_pairs_per_date(previous_pairs, d)
        backward, forward, bonus = match_pair(d, dates, target, bt_min, bt_max, count)
        if backward is not None:
            pairs.append(find_pair_from_dates(backward.date1, backward.date2, bpairs))
        if forward is not None:
            pairs.append(find_pair_from_dates(forward.date1, forward.date2, bpairs))
        if bonus is not None:
            pairs.append(find_pair_from_dates(bonus.date1, bonus.date2, bpairs))

    print("Writing {} pairs: {}".format(len(pairs), table))
    write_table(table, pairs)
