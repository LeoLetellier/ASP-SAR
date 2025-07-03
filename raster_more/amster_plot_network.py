#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster_plot_network.py
------------

Usage: amster_plot_network.py <table> <baselines>

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
import pydot


def open_table(file):
    content = np.genfromtxt(file, skip_header=1, usecols=(0, 1), dtype=str)
    return content


def open_baselines(file):
    content = np.genfromtxt(file, usecols=(0, 1, 2, 3), dtype=str)
    return content


def plot_network(pairs, baselines):
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    graph = pydot.Dot("interferogram_network", graph_type="digraph")

    for p in pairs:
        graph.add_edge(pydot.Edge(p[0].strip(), p[1].strip()))
    for b in baselines:
        graph.add_node(pydot.Node(b[0].strip(), label=b[0].strip(), bperp=float(b[1])))
    x, y = [], []
    for n in graph.get_nodes():
        x.append(date2num(datetime.strptime(n.get_label(), "%Y%m%d")))
        y.append(float(n.get_attributes()["bperp"]))
    ax.plot(x, y, "o", color='dodgerblue', mec='black', markersize=4, picker=5)
    for edge in graph.get_edges():
        if graph.get_node(edge.get_source()) is not None and graph.get_node(edge.get_destination()) is not None:
            master = graph.get_node(edge.get_source())[0]
            slave = graph.get_node(edge.get_destination())[0]
            x = date2num(datetime.strptime(master.get_label(), "%Y%m%d"))
            y = float(master.get_attributes()["bperp"])
            dx = date2num(datetime.strptime(slave.get_label(), "%Y%m%d")) - x
            dy = float(slave.get_attributes()["bperp"]) - y
            ax.arrow(x, y, dx, dy, linewidth=.5, color='black', alpha=.5)
    # an = ax.annotate("",
    #              xy=(0, 0), xycoords="data",
    #              xytext=(0, 0), textcoords="data",
    #              arrowprops=dict(facecolor="black", width=1, frac=0.3),
    #              bbox=dict(boxstyle="round", fc="w"))
    # an.set_visible(False)
    fig.suptitle("Interferogram network")
    fig.autofmt_xdate()
    ax.set_xlabel("Acquisition Date")
    ax.set_ylabel("Perpendicular Baseline (m)")
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y/%m/%d"))


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    table = arguments["<table>"]
    baselines_file = arguments["<baselines>"]

    pairs = open_table(table)
    print("pairs", pairs)
    baselines = open_baselines(baselines_file)
    print("baselines", baselines)

    all_dates = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))

    primary = baselines[0][0]
    if primary not in baselines[1]:
        primary = baselines[0][1]
    print("primary", primary)
    print(len(all_dates))

    bb = []
    if primary in all_dates:
        bb.append([primary, 0])
    for b in baselines:
        date = b[0]
        bp = str(-float(b[2]))
        if date == primary:
            date = b[1]
            bp = b[2]
        else:
            print("switch")
        if date in all_dates:
            bb.append([date, bp])

    print(all_dates)
    print(bb)

    if len(all_dates) != len(bb):
        print("length pb")

    plot_network(pairs, bb)
    plt.show()
