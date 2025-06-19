#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster2aspsar.py
--------------
Prepare the directory structure for further processing. Link all ALL2GIF results in the given destination dir.

Usage: amster2aspsar.py --amster=<path> --aspsar=<path> [--s1]
amster2aspsar.py -h | --help

Options:
-h | --help         Show this screen
--amster              Path to ALL2GIF results
--aspsar              Path to destination directory. Will be the processing directory
--s1

"""

from check_all2gif import check_for_empty_files
from prepare_correl_dir import prepare_dir_list, link_files, get_az_range_sampling
from convert_geotiff import convert_all

import os
import docopt
import pandas as pd


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    sm_dir = arguments["--amster"]
    working_dir = arguments["--aspsar"]
    s1 = arguments["--s1"]

    if len(check_for_empty_files(sm_dir)) >= 1:
        raise Exception('Check ALL2GIF directories')
    
    data_path_list = prepare_dir_list(sm_dir)
    link_files(data_path_list, working_dir)

    geotiff_dir = os.path.join(working_dir, "GEOTIFF")
    correl_dir = os.path.join(working_dir, "CORREL")

    try:
        os.mkdir(geotiff_dir)
    except:
        print("GEOTIFF dir already exists, keeping it")
        
    try:
        os.mkdir(correl_dir)
    except:
        print("CORREL dir already exists, keeping it")

    get_az_range_sampling(data_path_list, working_dir)

    all_file_df = pd.DataFrame(columns=['file', 'ncol', 'nrow'])
    convert_all(working_dir, all_file_df, geotiff_dir, s1)
    