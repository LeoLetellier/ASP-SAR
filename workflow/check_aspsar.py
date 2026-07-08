#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_aspsar.py
_______________

Usage: check_aspsar.py <aspsar> [--pairs=<pairs>]
check_aspsar.py -h | --help

Options:
  -h --help       Show this screen
  --pairs=<pairs> Path to the pairs file [default: <aspsar>/PAIRS/table_pairs.txt]
"""

import os
import logging
from glob import glob
import numpy as np
from docopt import docopt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_pair_list(pair_list_path: str) -> np.ndarray | None:
    """Load and return pairs from the pair list file."""
    try:
        pairs = np.loadtxt(pair_list_path, skiprows=2, usecols=(0, 1), dtype=str)
        return pairs
    except Exception as e:
        logging.error(f"Failed to load pairs from {pair_list_path}: {e}")
        return None

def pairs2dates(pairs: np.ndarray) -> list[str]:
    """Convert pairs to a sorted list of unique dates."""
    dates = []
    for p in pairs:
        dates.extend([p[0], p[1]])
    dates = list(set(dates))
    dates.sort()
    return dates

def check_geotiff(dates: list[str], folder: str) -> tuple[list[str], list[str]]:
    """Check for missing and unused geotiff files."""
    detected = glob(os.path.join(folder, "*.tif"))
    detected_dates = [os.path.basename(d).replace('.', '_').split('_')[0] for d in detected]

    missing_geotiff = [d for d in dates if d not in detected_dates]
    not_used = [d for d in detected_dates if d not in dates]

    return missing_geotiff, not_used

def check_stereo_results(pairs: np.ndarray, folder: str) -> tuple[list[str], list[str]]:
    """Check for missing and unused stereo results."""
    detected = glob(os.path.join(folder, "*", "*-F.tif"))
    detected_pairs = [os.path.basename(os.path.dirname(d)).split('_') for d in detected]

    missing_pair = [tuple(p) for p in pairs if list(p) not in detected_pairs]
    not_used = [p for p in detected_pairs if p not in pairs.tolist()]

    return missing_pair, not_used

def check_postproc():
    """Placeholder for post-processing checks."""
    pass

if __name__ == "__main__":
    arguments = docopt(__doc__)
    directory = arguments["<aspsar>"]
    pair_list_path = arguments["--pairs"] or os.path.join(directory, "PAIRS", "table_pairs.txt")

    pairs = check_pair_list(pair_list_path)
    if pairs is None:
        raise SystemExit("Failed to load pairs. Exiting.")

    dates = pairs2dates(pairs)

    geotiff_folder = os.path.join(directory, "GEOTIFF")
    stereo_folder = os.path.join(directory, "STEREO")

    missing_geotiff, unused_geotiff = check_geotiff(dates, geotiff_folder)
    missing_stereo, unused_stereo = check_stereo_results(pairs, stereo_folder)

    if missing_geotiff:
        print("Missing geotiffs:", ", ".join(missing_geotiff))
    if unused_geotiff:
        print("Geotiffs not used:", ", ".join(unused_geotiff))
    if missing_stereo:
        print("Missing correlation:", ", ".join([f"{p[0]}_{p[1]}" for p in missing_stereo]))
    if unused_stereo:
        print("Correlation not used:", ", ".join([f"{p[0]}_{p[1]}" for p in unused_stereo]))

    print("\n\tdone.")