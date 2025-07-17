#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_nsbas_process.py
---------------
Prepare the necessary files and directory structure for the NSBAS time series processing.

Usage: prepare_nsbas_process.py --data=<path> [--masked]
prepare_nsbas_process.py -h | --help

Options:
-h | --help         Show this screen
--data              Path to working directory to prepare
--masked            Use the masked files to prepare NSBAS processing

"""

import os, sys
import glob
import numpy as np
from osgeo import gdal
import pandas as pd
from pathlib import Path
import shutil
from dateutil import parser
import docopt


def date_to_float(d):
    """convert date to float for NSBAS processing (PyGdalSAR/correl/prep_correl_invers_pixel.py)"""
    return (d.year + (d.month-1)/12.0 + (d.day-1)/365.0)


def get_dates_and_Bp(pair_table):
    """
    help function to read data
    
    input is table_... created with Prepa_MSBAS.sh (Master)
    """
    # skip first two rows when reading data because of structure of table_...txt
    # needs to be adjusted if different input table file is used
    # pair_df = pd.read_csv(pair_table, sep='\t', header=None, skiprows=2)
    #print(pair_df)
    # dates1, dates2, bp = pair_df.iloc[:,0].to_list(), pair_df.iloc[:,1].to_list(), pair_df.iloc[:,2].to_list()
    dates1, dates2, bp = np.loadtxt(pair_table, unpack=True, skiprows=2, usecols = (0, 1, 2))
    return (dates1, dates2, bp) 


def get_perp_baseline_each_date(pair_table, date_list_file):
    """calculate the Bp for each date using the table_0....txt(Master) and date_list(prepare_result_export) file """
    # load bp for each pair and dates1 + dates2
    dates1, dates2, bp = get_dates_and_Bp(pair_table)
    M = len(bp)

    # load list of dates
    date_list = pd.read_csv(date_list_file, header=None).iloc[:,0].to_list()
    N = len(date_list)

    # build G
    G = np.zeros((M,N))
    
    for k in range((M)):
        for n in range((N)):
            if(dates1[k] == date_list[n]):
                G[k, n] = -1
            if(dates2[k] == date_list[n]):
                G[k, n] = 1

    # set first column of G to 0 -> get first date as reference; Bp of first date = 0
    G[:,0] = 0
    m = np.linalg.lstsq(G, bp, rcond=-1)

    return list(m[0])


def generate_list_pair(process_orient_dir, pair_table):
    """generate list_pair file and saves it in /NSBAS_PROCESS/H|V"""
    # get only master and slave dates - keep as pairs
    # pairs= pd.read_csv(pair_table, sep='\t', dtype=str).iloc[:,0:2]
    # pairs.to_csv(os.path.join(process_orient_dir, 'list_pair'), sep='\t', header=False, index=False)

    pairs = np.loadtxt(pair_table, usecols=(0,1), skiprows=2, dtype=str)
    print(pairs)
    np.savetxt(os.path.join(process_orient_dir, 'list_pair'), pairs, delimiter='\t', fmt='%s') 
    

def generate_list_dates(process_orient_dir, date_list_file, pair_table):
    """
    for first try, run with Bp=0 for each date -> add calculation later with get_perp_baseline_each_date

    generate list_dates file and saves it in /NSBAS_PROCESS/H|V
    """
    # read all dates in DataFrame structure (just one column)
    date_list = pd.read_csv(date_list_file, header=None).iloc[:,0].to_list()
    # convert all dates to decimal
    date_dec_list = [date_to_float(parser.parse(str(d))) for d in date_list]
    
    # set ref date -> everything relative to first date
    ref_date = date_dec_list[0]

    # calculate Bt
    date_diff = [d_dec - ref_date for d_dec in date_dec_list]

    # calculates Bp for each date with first date as reference
    bp = get_perp_baseline_each_date(pair_table, date_list_file)

    # prepare DataFrame for output
    out_df = pd.DataFrame({
        'dates': date_list,
        'dec': date_dec_list,
        'diff': date_diff,
        'bp': bp
        })

    out_df.to_csv(os.path.join(process_orient_dir, 'list_dates'), sep=' ', header=False, index=False)


def prepare_process_directories(nsbas_input_dir, nsbas_process_path, orientation, pair_table, date_list_file, masked):
    # orientation is 'H' or 'V'
    # if masked data is used, nsbas_input_dir = NSBAS/MASKED
    # if masked data is used, nsbas_process_path = NSBAS_PROCESS/MASKED/H|V
    # create subdir in NSBAS_PROCESS based on orientation
    process_orient_dir = os.path.join(nsbas_process_path, orientation)
    Path(process_orient_dir).mkdir(parents=True, exist_ok=True)

    # copy the input_inv_send in each dir
    shutil.copy(os.path.join(nsbas_input_dir, 'input_inv_send'), os.path.join(process_orient_dir, 'input_inv_send'))

    # generate list_pair based on table_... (created with PrepaMSBAS)
    generate_list_pair(process_orient_dir, pair_table)

    # generate list_dates
    generate_list_dates(process_orient_dir, date_list_file, pair_table)


    if(masked):
        input_orient_dir = os.path.join(nsbas_input_dir, 'MASKED', orientation) 
    # get /EXPORT/NSBAS/orientation dir
    else:
        input_orient_dir = os.path.join(nsbas_input_dir, orientation)

    # create NSBAS_PROCESS/orientation/LN_DATA dir
    ln_data_dir = os.path.join(process_orient_dir, 'LN_DATA')
    Path(ln_data_dir).mkdir(parents=True, exist_ok=True)

    # generates links in /orientation/LN_DATA to .r4 and .r4.rsc files
    for f in os.listdir(input_orient_dir):
        # need to put string into specific format
        # is: DATE1-DATE2_DIRECTION.r4/.rsc; need: DATE1-DATE2.r4/.rsc
        # need to check extensions because otherwise -> two with same name
        if(len(f.split('.')) == 3):
            ext = '.r4.rsc'
        else:
            ext = '.r4'
        if(os.path.isfile(os.path.join(ln_data_dir, '{}{}'.format(f.split('_')[0], ext)))):
            continue
        else:
            os.symlink(os.path.join(input_orient_dir, f), os.path.join(ln_data_dir, '{}{}'.format(f.split('_')[0], ext)))

    # copy the input_inv_send in each dir
    #shutil.copy(os.path.join(nsbas_input_dir, 'input_inv_send'), os.path.join(process_orient_dir, 'input_inv_send'))

    # generate list_pair based on table_... (created with PrepaMSBAS)
    #generate_list_pair(process_orient_dir, pair_table)

    # generate list_dates
    #generate_list_dates(process_orient_dir, date_list_file, pair_table)
    

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    # path to data dir (former  working directory)
    work_dir = arguments['--data']

    # check if masked is set
    masked = arguments['--masked']

    if(masked):
        nsbas_process_dir = os.path.join(work_dir, 'NSBAS_PROCESS', 'MASKED')
        Path(nsbas_process_dir).mkdir(parents=True, exist_ok=True)
    else:
        nsbas_process_dir = os.path.join(work_dir, 'NSBAS_PROCESS')
        Path(nsbas_process_dir).mkdir(parents=True, exist_ok=True)

    nsbas_input_dir = os.path.join(work_dir, 'EXPORT', 'NSBAS')

    # TODO: check this part again - maybe give as parameter
    # correl_dir = os.path.join(work_dir, 'CORREL')
    # get table file -> must be th only table_[...].txt file
    pair_table = os.path.join(work_dir, "PAIRS", "table_pairs.txt")
    # pair_table = [os.path.join(correl_dir, f) for f in os.listdir(correl_dir) if os.path.isfile(os.path.join(work_dir, f)) and f.split('_')[0] == 'table'][0]
    date_list_file = os.path.join(nsbas_input_dir, 'dates_list.txt')


    print('START PREPARING H DIRECTORY')
    prepare_process_directories(nsbas_input_dir, nsbas_process_dir, 'H', pair_table, date_list_file, masked)
    print('FINISHED H')

    print('START PREPARING V DIRECTORY')
    prepare_process_directories(nsbas_input_dir, nsbas_process_dir, 'V', pair_table, date_list_file, masked)
    print('FINISHED V')
