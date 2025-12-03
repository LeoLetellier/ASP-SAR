#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sub_pairs.py
------------

Usage: sub_pairs.py <table> [--all-pairs=<all-pairs>]

Options:
-h | --help         Show this screen
"""

import docopt
import os
import numpy as np
from tqdm import tqdm


class Pair:
    def __init__(self, d1, d2, bp, bt):
        self.date1 = d1
        self.date2 = d2
        self.bperp = float(bp)
        self.btemp = float(bt)
        self.is_wet = self.has_date_wet(d1, d2, ["06", "07", "08", "09", "10"])

    def __eq__(self, value):
        return self.date1 == value.date1 and self.date2 == value.date2 and self.bperp == value.bperp and self.btemp == value.btemp
    
    def __hash__(self):
        return hash((self.date1, self.date2, self.bperp, self.btemp))
    
    def __format__(self, format_spec):
        return self.date1 + "\t" + self.date2
    
    @staticmethod
    def has_date_wet(d1, d2, wet_period):
        d1_wet = False
        d2_wet = False
        if d1[4:6] in wet_period:
            d1_wet = True
        if d2[4:6] in wet_period:
            d2_wet = True
        return (d1_wet, d2_wet)



def read_all_pairs(all_pairs):
    all_pairs_data = np.genfromtxt(all_pairs, skip_header=7, dtype=str)
    return all_pairs_data


def construct_pair(all_pairs_content, date1, date2):
    for apc in all_pairs_content:
        if apc[0] == date1 and apc[1] == date2:
            return Pair(date1, date2, apc[7], apc[8])


def open_table(file):
    content = np.genfromtxt(file, skip_header=1, usecols=(0, 1), dtype=str)
    return content


def save_sub_table(file, pairs):
    content = "#Date1\tDate2\tDate1Wet\tDate2Wet\n"
    for p in pairs:
        content += p.date1 + '\t' + p.date2 + '\t' + str(p.is_wet[0]) + '\t' + str(p.is_wet[1]) + '\n'
    with open(file, 'w') as outfile:
        outfile.write(content)
    print("Wrote", file)
    

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    table = arguments["<table>"]
    all_pairs = arguments["--all-pairs"]
    table_name = os.path.splitext(table)[0]

    table_content = open_table(table)
    all_pairs_content = read_all_pairs(all_pairs)
    pairs = []
    dates = []
    for p in tqdm(range(len(table_content))):
        d1 = table_content[p][0]
        d2 = table_content[p][1]
        dates.append(d1)
        dates.append(d2)
        associated_pair = construct_pair(all_pairs_content, d1, d2)
        pairs.append(associated_pair)
    
    dates = list(set(dates))

    # Direct: two consecutive dates
    direct_pairs = []
    for p in pairs:
        id = dates.index(p.date1)
        if id != len(dates) - 1:
            if dates[id + 1] == p.date2:
                direct_pairs.append(p)
    save_sub_table(table_name + "_direct.txt", direct_pairs)

    # Short: less than 1 months
    short_pairs = []
    for p in pairs:
        if p.btemp < 32:
            short_pairs.append(p)
    save_sub_table(table_name + "_short.txt", short_pairs)

    # Long: more than 5 months
    long_pairs = []
    for p in pairs:
        if p.btemp > 150:
            long_pairs.append(p)
    save_sub_table(table_name + "_long.txt", long_pairs)

    # 3 Months (plus-minus tolerance)
    three_months_pairs = []
    for p in pairs:
        if p.btemp > 75 and p.btemp < 105:
            three_months_pairs.append(p)
    save_sub_table(table_name + "_3months.txt", three_months_pairs)

    # 6 Months (plus-minus tolerance)
    six_months_pairs = []
    for p in pairs:
        if p.btemp > 165 and p.btemp < 195:
            six_months_pairs.append(p)
    save_sub_table(table_name + "_6months.txt", six_months_pairs)

    # Yearly (modulo 1 year)
    yearly_pairs = []
    for p in pairs:
        btemp_mod = p.btemp % 365
        if btemp_mod > 350 or (btemp_mod < 15 and p.btemp > 360):
            yearly_pairs.append(p)
    save_sub_table(table_name + "_yearly.txt", yearly_pairs)
