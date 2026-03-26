#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deramp_ts.py
------------

Usage: deramp_ts.py <files-pattern> --ramp=<ramp> [--date-offset=<offset>] [--ramp-opts=<opts>] [--force]

Options:
  -h, --help                Show this screen
  <files-pattern>           Pattern of the files to find, must be somewhat *_YYYYMMDD_YYYYMMDD* or at least a uid
  --ramp=<ramp>             Name of the ramp to use [check deramp_ransac.py]
  --date-offset=<offset>    Offset where the first date can be found, relative to '_' character
  --force                   Recompute the ramp estimation, otherwise use existing when available
"""

import docopt
import os, sys
import subprocess
import numpy as np
import glob
import logging

logger = logging.getLogger()


def sh(cmd: str, shell: bool = True):
    return subprocess.run(
        cmd,
        shell=shell,
        env=os.environ,
    )


def txt_count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


def txt_read_line(filename, line_number):
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == line_number:
                return line.strip()
    return None


def estimate_ramp_coefficients(file, ramp_name, opts="", force=False):
    existings = glob.glob(file + "_deramp.tif_coeffs_*.txt")
    existings.sort()
    if len(existings) == 0 or force:
        cmd = "deramp_ransac.py {} --outfile={}_deramp.tif --ramp={} --chunk-size=4096 --nosave-deramp --save-coeffs {}".format(
            file, file, ramp_name, opts
        )
        logger.info("> {}".format(cmd))
        sh(cmd)
    else:
        logger.info("Will use existing coeffs: {}".format(existings[-1]))


def read_coeffs_file(file):
    files = glob.glob(file + "_deramp.tif_coeffs_*.txt")
    files.sort()
    file = files[-1] # the wildcard is write date so take last
    coeffs_nb = txt_count_lines(file) - 2 # coeffs start at 3rd line
    coeffs = []
    for k in range(2, coeffs_nb + 2):
        line = txt_read_line(file, k)
        coeffs.append(float(line.replace(" ", "").replace("\t", "")[2:]))
    return coeffs


def inverse_coeffs_per_date(d1, d2, coeffs):
    input = "deramp_ts_coeffs_ramp_raw.txt"
    outfile = "deramp_ts_coeffs_ramp_date.txt"
    coeffs_as_lists = [list(row) for row in zip(*coeffs)]
    coeffs_nb = len(coeffs_as_lists)
    
    np.savetxt(input, np.column_stack([d1, d2] + coeffs_as_lists), "%s")

    cmd = "invert_pair2date.py --date1={},0 --date2={},1 --values={},{} --outfile={}".format(
        input, input, input, ','.join([str(2 + k) for k in range(coeffs_nb)]), outfile
    )
    logger.info("> {}".format(cmd))
    sh(cmd)

    return input, outfile, coeffs_nb


def backsimulate_coeffs(input, outfile, nb):
    backfile = "deramp_ts_coeffs_ramp_back.txt"
    
    cmd = "infer_date2pair.py --date={},0 --value={},{} --date1={},0 --date2={},1 --outfile={}".format(
        outfile, outfile, ','.join([str(1 + k) for k in range(nb)]), input, input, backfile
    )
    logger.info("> {}".format(cmd))
    sh(cmd)

    usecols = [0, 1] + [2 + k for k in range(nb)]
    data = np.loadtxt(backfile, usecols=usecols, unpack=True, dtype=str)
    
    return data[0], data[1], data[2:]

def apply_ramp_coeffs(file, ramp_name, coeffs, opts=""):
    cmd = "deramp_ransac.py {} --outfile={}_deramp.tif --ramp={} --chunk-size=4096 --coeffs={} {}".format(
        file, file, ramp_name, ",".join(coeffs), opts
    )
    logger.info("> {}".format(cmd))
    sh(cmd)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    logging.basicConfig(
                level=logging.INFO, format="%(levelname)s: %(asctime)s | %(message)s"
            )
    
    pattern = arguments["<files-pattern>"]
    ramp_name = arguments["--ramp"]
    offset = int(arguments["--date-offset"]) if arguments["--date-offset"] is not None else 0
    opts = arguments["--ramp-opts"] if arguments["--ramp-opts"] is not None else ""
    force = arguments["--force"]

    files = glob.glob(pattern)
    names = [os.path.basename(f).split('.')[0] for f in files]
    logger.info("Found {} files: {}".format(len(files), names))

    d1 = [f.split('_')[offset] for f in names]
    d2 = [f.split('_')[offset + 1] for f in names]

    dates = list(set(d1 + d2))
    dates.sort()

    logger.info("Found {} dates: {}".format(len(dates), dates))

    coeffs = []
    for f in files:
        estimate_ramp_coefficients(f, ramp_name, opts, force)
        coeffs.append(read_coeffs_file(f))
        logger.info("Solved ramp for {} with coeffs {}".format(f, coeffs[-1]))

    input, outfile, nb = inverse_coeffs_per_date(d1, d2, coeffs)
    logger.info("Inverted coeffs in time series in files:\n{}\n{}".format(input, outfile))

    backd1, backd2, back_coeffs = backsimulate_coeffs(input, outfile, nb)
    logger.info("Backsimulate the ramps:\n{}\n{}\n{}".format(d1, d2, back_coeffs))

    coeffs_as_lists = [list(row) for row in zip(*back_coeffs)]
    
    for i, f in enumerate(files):
        for k in range(len(backd1)):
            if (backd1[k] == d1[i] and backd2[k] == d2[i]) or (backd2[k] == d1[i] and backd1[k] == d2[i]):
                logger.info("Applying {} ramp for {} with backprop coeffs {}".format(
                    f, ramp_name, coeffs_as_lists[k]
                ))
                if not os.path.isfile(f + "_deramp.tif") or force:
                    apply_ramp_coeffs(f, ramp_name, coeffs_as_lists[k], opts)
                else:
                    logger.info("Deramp raster exists: {}. Keeping.".format(f + "_deramp.tif"))
                break
