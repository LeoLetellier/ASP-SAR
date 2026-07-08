#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_pair.py
-------------

Usage: plot_pair.py --date1=<d> --date2=<d> --values=<v>


Options:
  -h --help               Show this screen.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import docopt
import cmcrameri.cm as cmc
from datetime import datetime


def plot_single_pair(d1_dt, d2_dt, val, cmap, norm):
    color = cmap(norm(val))
    bt = abs((d2_dt - d1_dt).days)
    plt.plot([d1_dt, d2_dt], [bt, bt], c=color)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    date1_file, date1_col = arguments["--date1"].split(",")
    date2_file, date2_col = arguments["--date2"].split(",")
    values_wanted = arguments["--values"].split(",")
    value_file, value_col = values_wanted[0], values_wanted[1:]

    date1 = np.loadtxt(date1_file, usecols=int(date1_col), dtype=str, unpack=True)
    date2 = np.loadtxt(date2_file, usecols=int(date2_col), dtype=str, unpack=True)
    values = np.loadtxt(value_file, usecols=[int(v) for v in value_col], dtype=float, unpack=True)

    dates_initial = list(set(date1.tolist() + date2.tolist()))
    dates_initial.sort()

    # Discard nan values
    if values.ndim == 1:
        mask = ~np.isnan(values)
    else:
        mask = ~np.isnan(values).any(axis=0)
    date1 = date1[mask]
    date2 = date2[mask]
    values = values[mask] if values.ndim == 1 else values[:, mask]

    print(date1.shape, date1)

    # If there is only one column for values need to simulate a 2D array
    if values.ndim == 1:
        values = np.array([values])

    ncoeffs = len(values)

    cmap = cmc.navia
    cmap = cmc.lajolla.reversed()
    norm = Normalize(vmin=np.min(values), vmax=np.max(values))

    plt.figure()
    for k in range(len(date1)):
        for i in range(ncoeffs):
            plot_single_pair(
                datetime.strptime(date1[k], "%Y%m%d"),
                datetime.strptime(date2[k], "%Y%m%d"),
                values[i][k],
                cmap,
                norm
            )
    plt.show()
