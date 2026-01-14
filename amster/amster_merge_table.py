#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster_generate_network.py
------------

Usage: amster_generate_network.py <table1> <table2> --outfile=<outfile>

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

    table1 = arguments["<table1>"]
    table2 = arguments["<table2>"]
    outfile = arguments["--outfile"]

    pairs1 = open_table(table1)
    pairs2 = open_table(table2)

    all_pairs = pairs1 + pairs2
    all_pairs = list(set(all_pairs))

    print("Merging tables into: {}".format(outfile))
    write_table(outfile, all_pairs)
