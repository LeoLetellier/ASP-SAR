#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster_generate_network.py
------------

Usage: amster_generate_network.py <table> --bp-range=<bp-range> --bt-range=<bt-range> --outfile=<outfile>

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
        self.bp = abs(bp)
        self.bt = abs(bt)

    def get_line(self) -> str:
        return "\n{}\t{}\t{}\t\t{}".format(self.date1, self.date2, self.bp, self.bt)
    
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


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    table1 = arguments["<table>"]
    lim_bt = [float(k) for k in arguments["--bt-range"].split(",", 2)]
    lim_bp = [float(k) for k in arguments["--bp-range"].split(",", 2)]
    outfile = arguments["--outfile"]

    print("Filtering pairs to keep ones with bt between {} and {} days, and bp between {} and {}m".format(
        lim_bt[0], lim_bt[1], lim_bp[0], lim_bp[1]
    ))

    pairs = open_table(table1)
    
    pairs_out = [p for p in pairs if (p.bt > lim_bt[0] and p.bt < lim_bt[1] and p.bp > lim_bp[0] and p.bp < lim_bp[1])]

    print("Keeping {} out of {} pairs".format(
        len(pairs_out),
        len(pairs)
    ))

    print("Saving table into: {}".format(outfile))
    write_table(outfile, pairs_out)
