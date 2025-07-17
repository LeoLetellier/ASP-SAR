#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_correl_dir.py
--------------
Prepare the directory structure for further processing. Link all ALL2GIF results in the given destination dir.

Usage: prepare_correl_dir.py --data=<path> --dest=<path> [--u] 
prepare_correl_dir.py -h | --help

Options:
-h | --help         Show this screen
--data              Path to ALL2GIF results
--dest              Path to destination directory. Will be the processing directory
--u                 Only update the links

"""

import os
import pandas as pd
from pathlib import Path
import docopt


def filter_mod_files(input_file):
    """Check if the mod file has not zero dimensions"""
    i12_path = os.path.dirname(os.path.dirname(os.path.realpath(input_file)))
    insar_param_file = os.path.join(i12_path, 'TextFiles', 'InSARParameters.txt')
    with open(insar_param_file, 'r') as f:
        lines = [''.join(l.strip().split('\t\t')[0]) for l in f.readlines()]
        jump_index = lines.index('/* -5- Interferometric products computation */')
        img_dim = lines[jump_index + 2: jump_index + 4]

    ncol, nrow = int(img_dim[0].strip()), int(img_dim[1].strip())
    return not (ncol == 0 or nrow == 0)


def prepare_dir_list(input_path):
    """Retrieve all valid mod files"""
    data_dirs_paths = [os.path.join(input_path, d, 'i12', 'InSARProducts') for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d, 'i12', 'InSARProducts'))]
    data_dirs_paths = list(set(data_dirs_paths))
    out_list = []
    
    for d in data_dirs_paths:
        for f in os.listdir(d):
            file = os.path.join(d, f)
            if(os.path.splitext(file)[1] == '.mod'):
                if file not in out_list and filter_mod_files(file):
                    out_list.append(file)
    return out_list


def link_files(data_path_list, dst_path):
    """Link the mod files to the working directory"""
    print('Start linking files')
    for p in data_path_list:
        target = os.path.join(dst_path, os.path.basename(p))
        if not os.path.islink(target):
            print("symlink", p, target)
            os.symlink(p, target)
    print('Finished linking files')


def save_az_range_sampling(data_path_list, correl_path):
    """Fetch the pixel dimension and save it to file"""
    print(data_path_list)
    result_dir = os.path.dirname(os.path.dirname(data_path_list[0]))
    insar_param_file = os.path.join(result_dir, 'TextFiles', 'InSARParameters.txt')
    with open(insar_param_file, 'r') as f:
        lines = [''.join(l.strip().split('\t\t')) for l in f.readlines()]
        for l in lines:
            if 'Azimuth sampling' in l:
                azimuth_sampl = l.split('/')[0].strip()
            elif 'Slant range sampling' in l:
                range_sampl = l.split('/')[0].strip()
            # also get ML factor to get final sampling
            elif '/* Range reduction factor [pix] */' in l:
                range_factor = l.split('/')[0].strip()
            elif '/* Azimuth reduction factor [pix] */' in l:
                azimuth_factor = l.split('/')[0].strip()

    sampling = pd.DataFrame(data={'AZ':[float(azimuth_sampl) * float(azimuth_factor)], 'SR':[float(range_sampl) * float(range_factor)]})
    sampling.to_csv(os.path.join(correl_path, 'sampling.txt'), sep='\t', index=None)


if __name__ == "__main__":
    from init_asp_parameters import init_asp_parameters 
    arguments = docopt.docopt(__doc__)

    # path to data processed with MasTer
    input_path = arguments['--data']
    # path to where data should be linked to; directory for further processing
    dst_path = arguments['--dest']
    # update flag
    update = arguments['--u']

    data_path_list = prepare_dir_list(input_path)

    if(update):
        print('Update directory')
        link_files(data_path_list, dst_path)
    else:
        print('Initiate directory')
        link_files(data_path_list, dst_path)

        Path(os.path.join(dst_path, 'GEOTIFF')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dst_path, 'CORREL')).mkdir(parents=True, exist_ok=True)
        correl_path = os.path.join(dst_path, 'CORREL')
    
        # save asp_parameters and sampling in destination directory instead of CORREL
        init_asp_parameters(dst_path)

        save_az_range_sampling(data_path_list, dst_path)
