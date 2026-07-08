#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export2nsbas.py
-----------
Prepare a NSBAS directory given an EXPORT directory where H, V, NCC can be found

Usage: export2nsbas.py <export> <nsbas> [--pairs=<pairs>] [--dates=<dates>] [--no-bp] [-v | --verbose]
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
        # date1 date2 bp bt
        pairs = [Pair(p[0], p[1], p[3], p[2]) for p in pairs]
        return pairs

    def get_dates(self):
        return [self.date1, self.date2]
    
    def get_dates_as_float(self):
        date1_dt = datetime.strptime(self.date1, "%Y%m%d")
        date2_dt = datetime.strptime(self.date2, "%Y%m%d")
        return [
            date2float(date1_dt),
            date2float(date2_dt)
            ]
    

def date2float(date):
    if type(date) in [str, np.str_]:
        date = datetime.strptime(date, "%Y%m%d")
    elif type(date) is int:
        date = datetime.strptime(str(date), "%Y%m%d")
    return date.year + (date.month - 1) / 12.0 + (date.day - 1) / 365.0


def write_nsbas_input_inv_send(nsbas_dir):
    content = """\
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
    """
    if not os.path.exists(os.path.join(nsbas_dir, 'V', "input_inv_send")):
        with open(os.path.join(nsbas_dir, 'V', "input_inv_send"), 'w') as f:
            f.write(content)
    if not os.path.exists(os.path.join(nsbas_dir, 'H', "input_inv_send")):
        with open(os.path.join(nsbas_dir, 'H', "input_inv_send"), 'w') as f:
            f.write(content)


def write_nsbas_list_pair(nsbas_dir, pairs):
    pairs = [p.get_dates() for p in pairs]

    np.savetxt(os.path.join(nsbas_dir, 'H', 'list_pair'), pairs, delimiter='\t', fmt='%s')
    np.savetxt(os.path.join(nsbas_dir, 'V', 'list_pair'), pairs, delimiter='\t', fmt='%s')
    logger.info(f"Wrote {len(pairs)} pairs entry to file")


def write_nsbas_list_date(nsbas_dir, pairs):
    list_date = format_nsbas_date(pairs)

    np.savetxt(os.path.join(nsbas_dir, 'H', 'list_dates'), list_date, delimiter='\t', fmt='%s')
    np.savetxt(os.path.join(nsbas_dir, 'V', 'list_dates'), list_date, delimiter='\t', fmt='%s')
    logger.info(f"Wrote {len(list_date)} date entry to file")


def format_nsbas_date(pairs):
    dates = []
    for p in pairs:
        dates += p.get_dates()
    dates = list(set(dates))
    dates.sort()

    logger.info(f"Got {len(dates)} unique dates")

    dates_dt = [datetime.strptime(d, "%Y%m%d") for d in dates]
    dates_dec = [d.year + (d.month - 1) / 12.0 + (d.day - 1) / 365.0 for d in dates_dt]
    dates_dec_datum = [d - dates_dec[0] for d in dates_dec]

    bp = get_daily_bp(pairs, dates)

    return np.column_stack([dates, dates_dec, dates_dec_datum, bp])


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
    N = len(dates)
    M = len(pairs)
    G = np.zeros((M, N))
    # print(G.shape, G.dtype)
    dates = [d for d in dates]
    
    dates1, dates2, bp = [p.date1 for p in pairs], [p.date2 for p in pairs], [float(p.bp) for p in pairs]

    print([k for k in zip(dates1, dates2, bp)])
    print(np.mean(bp), np.median(bp))

    for k in range((M)):
        for n in range((N)):
            if(dates1[k] == dates[n]):
                G[k, n] = -1
            if(dates2[k] == dates[n]):
                G[k, n] = 1
    
    G[:,0] = 0
    # print(len(bp), np.array(bp).dtype, bp)
    m = np.linalg.lstsq(G, bp, rcond=-1)
    # print(len(m), m)

    print(list(m[0]))
    print(np.mean(list(m[0])), np.median(list(m[0])))

    return list(m[0])


def test_get_daily_bp():
    dates = ["20250101", "20250102", "20250103", "20250104"]
    bp_dates = [0, 23, -12, 36]
    pairs = [
        Pair("20250101", "20250102", None, bp_dates[1] - bp_dates[0]),
        Pair("20250101", "20250103", None, bp_dates[2] - bp_dates[0]),
        Pair("20250101", "20250104", None, bp_dates[3] - bp_dates[0]),
        Pair("20250102", "20250103", None, bp_dates[2] - bp_dates[1]),
        Pair("20250102", "20250104", None, bp_dates[3] - bp_dates[1]),
        Pair("20250103", "20250104", None, bp_dates[3] - bp_dates[2]),
    ]
    daily_bp = get_daily_bp(pairs, dates)

    return np.allclose(bp_dates, daily_bp)


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


def link_data(nsbas, pairs):
    for p in pairs:
        d1, d2 = p.get_dates()

        def linker(source, target):
            if not os.path.islink(target):
                os.symlink("../../../" + source, target)
            if not os.path.islink(target + ".rsc"):
                os.symlink("../../../" + source + ".rsc", target + ".rsc")

        
        linker(p.path_v, os.path.join(nsbas, "V", "LN_DATA", d1 + "-" + d2 + ".r4"))
        linker(p.path_h, os.path.join(nsbas, "H", "LN_DATA", d1 + "-" + d2 + ".r4"))
        linker(p.path_ncc, os.path.join(nsbas, "V", "LN_DATA", d1 + "-" + d2 + "-CC.r4"))
        linker(p.path_ncc, os.path.join(nsbas, "H", "LN_DATA", d1 + "-" + d2 + "-CC.r4"))


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

    logger.info(f"Processing {len(pairs)} pairs")

    for p in pairs:
        p.get_expected_paths(export_dir)
        if not all(p.check_files_exists()):
            raise FileNotFoundError(f"Not found: {p.path_v}, {p.path_h}")

    def ensure_dir(dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
    
    ensure_dir(os.path.join(nsbas_dir, "H"))
    ensure_dir(os.path.join(nsbas_dir, "H", "LN_DATA"))
    ensure_dir(os.path.join(nsbas_dir, "V"))
    ensure_dir(os.path.join(nsbas_dir, "V", "LN_DATA"))

    write_nsbas_list_pair(nsbas_dir, pairs)
    write_nsbas_list_date(nsbas_dir, pairs)
    write_nsbas_input_inv_send(nsbas_dir)
    link_data(nsbas_dir, pairs)
