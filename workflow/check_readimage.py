#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_readimage.py
-----------
Check if all images / dates have been red by amster from SAFE

Usage: check_readimage.py <safe> <csl>
check_readimage.py -h | --help

Options:
-h --help       Show this screen

"""

import os
import docopt
import glob


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    safe = arguments["<safe>"]
    csl = arguments["<csl>"]

    files_safe = glob.glob(safe + "/*.SAFE")
    if os.path.isdir(safe + "/FORMER"):
        files_safe += glob.glob(safe + "/FORMER/*/*.SAFE")
    
    dates_safe = [f.split('_')[5][:8] for f in files_safe]
    dates_safe = list(set(dates_safe))
    dates_safe.sort()

    print("Dates in SAFE (Raw):")
    print(' '.join(dates_safe))

    files_csl = glob.glob(csl + "/*.csl")

    dates_csl = [f.split('_')[3] for f in files_csl]
    dates_csl = list(set(dates_csl))
    dates_csl.sort()

    print("Dates in CSL (AMSTer):")
    print(' '.join(dates_csl))

    safe_not_in_csl = []
    csl_not_in_safe = []

    for d in dates_safe:
        if d not in dates_csl:
            safe_not_in_csl.append(d)
    
    for d in dates_csl:
        if d not in dates_safe:
            csl_not_in_safe.append(d)

    if len(safe_not_in_csl) > 0 or len(csl_not_in_safe) > 0:
        print("\nWarning: inconsistencies were found between SAFE and CSL folders\n")

    if len(safe_not_in_csl) > 0:
        print("Found {} dates in SAFE but not red by AMSTer: {}".format(
            len(safe_not_in_csl),
            " ".join(safe_not_in_csl)
        ))

    if len(csl_not_in_safe) > 0:
        print("Found {} dates red by AMSTer but not in current SAFE folder: {}".format(
            len(csl_not_in_safe),
            " ".join(csl_not_in_safe)
        ))

    csl_no_info = []

    for dcsl in dates_csl:
        file_info = glob.glob(csl + "/*" + dcsl + "*.csl/Info/SLCImageInfo.txt")
        if len(file_info) == 0:
            csl_no_info.append(dcsl)
    
    if len(csl_no_info) > 0:
        print("Found {} dates where no info is available in AMSTer: {}".format(
            len(csl_no_info),
            " ".join(csl_no_info)
        ))
