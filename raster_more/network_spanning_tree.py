#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_spanning_tree.py
------------

Usage: network_spanning_tree.py <pairs> --outfile=<outfile> [--w-col=<w-col>]

Options:
-h | --help         Show this screen
<pairs>             Column file with date1 date2 Bt Bp Weight
"""

import docopt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def build_graph(edges):
    G = nx.Graph()
    for date1, date2, weight in edges:
        G.add_edge(date1, date2, weight=weight)
    return G


def visualize_mst(mst):
    pos = nx.spring_layout(mst)
    nx.draw(mst, pos, with_labels=True, node_size=700, node_color='skyblue')
    labels = nx.get_edge_attributes(mst, 'weight')
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=labels)
    plt.show()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    pairs = arguments["<pairs>"]
    outfile = arguments["--outfile"]
    w_col = int(arguments["--w-col"]) if arguments["--w-col"] is not None else 2

    date1, date2 = np.loadtxt(pairs, usecols=(0, 1), dtype=str, unpack=True)
    weight = np.loadtxt(pairs, usecols=(w_col), unpack=True)
    edges = [(date1[k], date2[k], weight[k]) for k in range(len(date1))]

    G = build_graph(edges)
    tree = nx.minimum_spanning_tree(G)
    visualize_mst(tree)
