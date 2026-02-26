#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_weights.py
------------

Usage: network_weights.py <infile>

Options:
-h | --help         Show this screen
"""

import docopt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Pair:
    def __init__(self, date1, date2, bp=None):
        if date1 > date2:
            self.date1 = datetime.strptime(date2, "%Y%m%d")
            self.date2 = datetime.strptime(date1, "%Y%m%d")
        else:
            self.date1 = datetime.strptime(date1, "%Y%m%d")
            self.date2 = datetime.strptime(date2, "%Y%m%d")
        self.bt = self._compute_bt()
        self.bp = None

    def _compute_bt(self):
        self.bt = (self.date2 - self.date1).days
        return self.bt
    
    def __eq__(self, value):
        return self.date1 == value.date1 and self.date2 == value.date2
    
    def __lt__(self, other):
        if self.date1 == other.date1:
            return self.date2 <= other.date2
        return self.date1 <= other.date1
    
    @staticmethod
    def from_file_list(file, index=None):
        date1, date2 = np.loadtxt(file, usecols=(0, 1), unpack=True, dtype=str)
        bt, bp = np.loadtxt(file, usecols=(2, 3), unpack=True)
        pairs = []
        for k in range(len(date1)):
            pairs.append(Pair(date1[k], date2[k], bp[k]))
        return pairs
    
    def weight(self, target_dates, bt_range, bp_range, do_less_bp, do_date_anniversary, dtp_power):
        pass
        
        return 


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    file = arguments["<infile>"]

    pairs = Pair.from_file_list(file)
