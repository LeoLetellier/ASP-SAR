#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster_table_add.py
------------

Usage: amster_table_add.py <dates>... [--table=<table>] [--all-pairs=<all-pairs>]

Options:
-h | --help         Show this screen
"""

import docopt
import numpy as np


def fetch_baselines(all_pairs, date1, date2):
    data = np.genfromtxt(all_pairs, skip_header=7, dtype=str)
    data = [d for d in data if date1 in d and date2 in d][0]
    return data[7], data[8]


def append_table(table, date1, date2, bp, bt):
    new_line = f"{date1}\t{date2}\t{bp}\t\t{bt}"
    print("adding:\t", new_line)
    with open(table, 'a') as t:
        t.write(new_line)
        

def add_dates(dates, table, all_pairs):
    for d1, d2 in dates:
        print("Processing:", d1, d2)
        if int(d1) > int(d2):
            d1, d2 = d2, d1
        bp, bt = fetch_baselines(all_pairs, d1, d2)
        append_table(table, d1, d2, bp, bt)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    dates = arguments["<dates>"]
    table = arguments["--table"]
    all = arguments["--all-pairs"]

    dates = [d.split('-', 2) for d in dates]
    
    if table is None:
        table = "table_pairs.txt"
    if all is None:
        all = "allPairsListing.txt"
    
    add_dates(dates, table, all)
