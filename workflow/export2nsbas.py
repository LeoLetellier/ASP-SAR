#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export2nsbas.py
-----------
Prepare a NSBAS directory given an EXPORT directory where H, V, NCC can be found

Usage: export2nsbas.py <export> <nsbas> [--pairs=<pairs>] [--dates=<dates>] --no-bp [-v | --verbose]
export2nsbas.py -h | --help

Options:
  -h, --help       Show this screen
  -v, --verbose
"""

import os
import docopt
import numpy as np
import glob
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Pair:
    def __init__(self, date1, date2, bt=None, bp=None):
        self.date1 = date1
        self.date2 = date2
        self.bt = bt
        if bt is None:
            self.bt = (datetime.strptime(date2, "%Y%m%d") - datetime.strptime(date1, "%Y%m%d")).days
        self.bp = bp
        if bp is None:
            self.bp = 0
        self.path_v = None
        self.path_h = None
        self.path_ncc = None
    
    def get_expected_paths(self, export_dir):
        self.path_v = os.path.join(
            export_dir,
            "V",
            "V_" + str(self) + ".r4"
        )
        self.path_h = os.path.join(
            export_dir,
            "H",
            "H_" + str(self) + ".r4"
        )
        self.path_ncc = os.path.join(
            export_dir,
            "NCC",
            "NCC_" + str(self) + ".r4"
        )
        return self
    
    def check_files_exists(self):
        h_exists = self.path_h is not None and os.path.isfile(self.path_h)
        v_exists = self.path_v is not None and os.path.isfile(self.path_v)
        ncc_exists = self.path_ncc is not None and os.path.isfile(self.path_ncc)
        return h_exists, v_exists, ncc_exists

    def __str__(self):
        return "{}_{}".format(self.date1, self.date2)
    
    def has_all(self):
        return self.has_h and self.has_v and self.has_ncc

    @staticmethod
    def read_from_file(table) -> list:
        pairs = np.loadtxt(table, usecols=(0, 1, 2, 3), skiprows=2, dtype=str)
        pairs = [Pair(p[0], p[1], p[2], p[3]) for p in pairs]
        return pairs

    def get_dates(self):
        return [self.date1, self.date2]
    
    def get_dates_as_float(self):
        date1_dt = datetime.strptime(self.date1, "%Y%m%d")
        date2_dt = datetime.strptime(self.date2, "%Y%m%d")
        return [
            date1_dt.year + (date1_dt.month - 1) / 12.0 + (date1_dt.day - 1) / 365.0, 
            date2_dt.year + (date2_dt.month - 1) / 12.0 + (date2_dt.day - 1) / 365.0
            ]


def write_nsbas_input_inv_send(nsbas_dir):
    with open(os.path.join(nsbas_dir, 'V', "input_inv_send"), 'w') as f:
        f.write("""\
0.03  #  temporal smoothing weight, gamma liss **2 (if <0.0001, no smoothing)
1     #   mask pixels with large RMS misclosure  (y=0;n=1)
1.5    #  threshold for the mask on RMS misclosure (in same unit as input files)
1      #  range and azimuth downsampling (every n pixel)
0      #  iterations to correct unwrapping errors (y:nb_of_iterations,n:0)
5      #  iterations to weight pixels of interferograms with large residual? (y:nb_of_iterations,n:0)
0.5    #  Scaling value for weighting residuals (1/(res**2+value**2)) (in same unit as input files) (Must be approximately equal to standard deviation on measurement noise)
4      #  iterations to mask (tiny weight) pixels of interferograms with large residual? (y:nb_of_iterations,n:0)
8.     #  threshold on residual, defining clearly wrong values (in same unit as input files)
1      #  outliers elimination by the median (only if nsamp>1) ? (y=0,n=1)
list_dates
0      #  sort by date (0) ou by another variable (1) ?
list_pair
1     #  interferogram format (RMG : 0; R4 :1) (date1-date2_pre_inv.unw or date1-date2.r4)
3100.  #  include interferograms with bperp lower than maximal baseline
0      #  Weight input interferograms by coherence or correlation maps ? (y:0,n:1)
1      #  coherence file format (RMG : 0; R4 :1) (date1-date2.cor or date1-date2-CC.r4)
1      #  minimal number of interferams using each image
1      #  interferograms weighting so that the weight per image is the same (y=0;n=1)
0.8    #  maximum fraction of discarded interferograms
0      #  Would you like to restrict the area of inversion ?(y=1,n=0)
1 735 1500 1585  #  Give four corners, lower, left, top, right in file pixel coord
1      #  referencing of interferograms by bands (1) or corners (2) ? (More or less obsolete)
5      #  band NW -SW(1), band SW- SE (2), band NW-NE (3), or average of three bands (4) or no referencement (5) ?
1      #  Weigthing by image quality (y:0,n:1) ? (then read quality in the list of input images)
1     #  Weigthing by interferogram variance (y:0,n:1) or user given weight (2)?
1      #  use of covariance (y:0,n:1) ? (Obsolete)
1      #  Adjust functions to phase history ? (y:1;n:0) Require to use smoothing option (smoothing coefficient) !
0      #  compute DEM error proportional to perpendicular baseline ? (y:1;n:0)
0 2003.0     #  include a step function ? (y:1;n:0)
0      #  include a cosinus / sinus function ? (y:1;n:0)
1      #  smoothing by Laplacian, computed with a scheme at 3pts (0) or 5pts (1) ?
2      #  weigthed smoothing by the average time step (y :0 ; n : 1, int : 2) ?
1      # put the first derivative to zero (y :0 ; n : 1)?
    """)


def write_nsbas_list_pair(nsbas_dir, pairs):
    pairs = [p.get_dates() for p in pairs]
    #TODO
    np.savetxt(os.path.join(nsbas_dir, 'H', 'list_pair'), pairs, delimiter='\t', fmt='%s')
    np.savetxt(os.path.join(nsbas_dir, 'V', 'list_pair'), pairs, delimiter='\t', fmt='%s')


def write_nsbas_list_date(nsbas_dir, dates):
    #TODO
    np.savetxt(os.path.join(nsbas_dir, 'H', 'list_dates'), dates, delimiter='\t', fmt='%s')
    np.savetxt(os.path.join(nsbas_dir, 'V', 'list_dates'), dates, delimiter='\t', fmt='%s')


def fetch_pairs(export_dir):
    h_pairs = glob.glob(os.path.join(export_dir, 'H', 'H_*_*.r4'))
    v_pairs = glob.glob(os.path.join(export_dir, 'V', 'V_*_*.r4'))
    ncc_pairs = glob.glob(os.path.join(export_dir, 'NCC', 'NCC_*_*.r4'))

    h_pairs_dates = [os.path.basename(p).split('_')[1:3] for p in h_pairs]
    v_pairs_dates = [os.path.basename(p).split('_')[1:3] for p in v_pairs]
    ncc_pairs_dates = [os.path.basename(p).split('_')[1:3] for p in ncc_pairs]

    full_pairs_dates = h_pairs_dates + v_pairs_dates + ncc_pairs_dates
    full_pairs_dates = list(set(full_pairs_dates))
    common_pairs = []
    for p in full_pairs_dates:
        if p in v_pairs_dates and p in h_pairs_dates and p in ncc_pairs_dates:
            common_pairs.append(p)
        else:
            logger.info("Pair {}-{} is not complete (H+V+NCC)".format(
                p[0],
                p[1]
            ))
    return common_pairs


def get_daily_bp(pairs, dates):
    pass


def infer_pair_from_files(folder):
    h_pairs = glob.glob(os.path.join(export_dir, 'H', 'H_*_*.r4'))
    v_pairs = glob.glob(os.path.join(export_dir, 'V', 'V_*_*.r4'))
    ncc_pairs = glob.glob(os.path.join(export_dir, 'NCC', 'NCC_*_*.r4'))

    h_pairs_dates = [os.path.basename(p).split('_')[1:3] for p in h_pairs]
    v_pairs_dates = [os.path.basename(p).split('_')[1:3] for p in v_pairs]
    ncc_pairs_dates = [os.path.basename(p).split('_')[1:3] for p in ncc_pairs]

    full_pairs_dates = h_pairs_dates + v_pairs_dates + ncc_pairs_dates
    full_pairs_dates = list(set(full_pairs_dates))
    pairs = []
    for k in range(len(full_pairs_dates)):
        pairs.append(Pair(full_pairs_dates[k][0], full_pairs_dates[k][1]))
    
    return pairs


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    if arguments["--verbose"]:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s: %(asctime)s | %(message)s"
        )

    export_dir = arguments["<export>"]
    nsbas_dir = arguments["<nsbas>"]
    pairs_file = arguments["--pairs"]
    no_bp = arguments["--no-bp"]

    if pairs_file is not None:
        pairs = Pair.read_from_file(pairs_file)
    else:
        pairs = infer_pair_from_files(export_dir)
    
    dates = []
    for p in pairs:
        dates += p.get_dates()
    dates = list(set(dates))
    dates.sort()
