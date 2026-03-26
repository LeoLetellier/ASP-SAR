#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""\
infer_date2pair.py
-------------

Usage: infer_date2pair.py --date=<date> --value=<value> --date1=<date1> --date2=<date2> --outfile=<outfile>
infer_date2pair.py  -h | --help

Options:
-h --help           Show this screen.
"""

import docopt
import numpy as np


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    date_file, date_col = arguments["--date"].split(",")
    date1_file, date1_col = arguments["--date1"].split(",")
    date2_file, date2_col = arguments["--date2"].split(",")
    values_wanted = arguments["--value"].split(",")
    value_file, value_col = values_wanted[0], values_wanted[1:]
    outfile = arguments["--outfile"]

    date = np.loadtxt(date_file, usecols=int(date_col), dtype=str, unpack=True)
    date1 = np.loadtxt(date1_file, usecols=int(date1_col), dtype=str, unpack=True)
    date2 = np.loadtxt(date2_file, usecols=int(date2_col), dtype=str, unpack=True)
    values = np.loadtxt(value_file, usecols=[int(v) for v in value_col], dtype=float, unpack=True)

    dates = list(set(date1.tolist() + date2.tolist()))
    dates.sort()
    date_sorted = date
    date_sorted.sort()
    for d in dates:
        if d not in date_sorted:
            raise ValueError('date mismatch')

    values_dics = [{} for _ in range(len(values))]
    for k in range(len(values)):
        for v, d in zip(values[k], date):
            values_dics[k][d] = v

    ndates = len(date)
    npairs = len(date1)

    if len(date2) != npairs or len(values[0]) != len(date):
        raise ValueError("size mismatch")

    inferred_values = [[] for _ in range(len(values))]

    for i in range(len(values)):
        values_dic = values_dics[i]
        for k in range(len(date1)):
            d1 = date1[k]
            d2 = date2[k]
            inferred_v = values_dic[d2] - values_dic[d1]
            inferred_values[i].append(inferred_v)

    print("Saving in the output file", outfile)
    # print(date1, date2, inferred_values)
    np.savetxt(outfile, np.vstack([date1, date2, inferred_values]).T, fmt="%s")
