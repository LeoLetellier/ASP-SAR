#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_network.py
------------

Usage: figure_network.py <table> [--out=<out>]

Options:
-h | --help         Show this screen
"""

import docopt
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

gdal.UseExceptions()


class Pair:
    def __init__(self, date1, date2):
        if date1 > date2:
            self.date1 = date2 = datetime.strptime(date2, "%Y%m%d")
            self.date2 = datetime.strptime(date1, "%Y%m%d")
        else:
            self.date1 = datetime.strptime(date1, "%Y%m%d")
            self.date2 = date2 = datetime.strptime(date2, "%Y%m%d")
        self.bt = None
        self._compute_bt()

    def _compute_bt(self):
        self.bt = (self.date2 - self.date1).days
        return self.bt


def open_table(file):
    content = np.loadtxt(file, usecols=(0, 1), dtype=str, delimiter="\t")
    return content


def write_table(file, pairs, weight):
    content = ""
    for p in range(len(pairs)):
        content += "\t".join([pairs[p][0], pairs[p][1], str(weight[p])]) + "\n"
    with open(file, 'w') as outfile:
        outfile.write(content)
    print("Saved:", file)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    table = arguments["<table>"]
    out = arguments["--out"]
    if out is None:
        out = os.path.splitext(table)[0] + "_with_weights.txt"
    
    pairs = open_table(table)
    weights = np.zeros(len(pairs))

    for p in range(len(pairs)):
        d1 = datetime.strptime(pairs[p][0], "%Y%m%d")
        d2 = datetime.strptime(pairs[p][1], "%Y%m%d")
        bt = (d2 - d1).days
        if bt <= 60:
            weights[p] = 1 / 8
        elif bt >= 365:
            weights[p] = 1
        else:
            weights[p] = (bt - 60) / 305 + (365 - bt) / 305 / 8
    
    write_table(out, pairs, weights)
