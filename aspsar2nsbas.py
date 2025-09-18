#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster2aspsar.py
--------------
Prepare the directory structure for further processing. Link all ALL2GIF results in the given destination dir.

Usage: amster2aspsar.py <aspsar> [--force] [--no-mask]
amster2aspsar.py -h | --help

Options:
-h | --help         Show this screen
--aspsar              Path to aspsar processing directory
--s1

"""

import docopt
import os
from os.path import join
from pathlib import Path
import shutil
from tqdm import tqdm

from workflow.prepare_result_export import *
from workflow.prepare_nsbas_process import *

DIRS = ['H', 'V', 'NCC']
EXPORT = 'EXPORT'


def open_gdal(path, band=1):
    ds = gdal.Open(path)
    ndv = ds.GetRasterBand(band).GetNoDataValue()
    data = ds.GetRasterBand(band).ReadAsArray()
    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    return data


def save_gdal(path, data, template, ndv):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, ds_template)
    band = ds.GetRasterBand(1)

    band_ndv = band.GetNoDataValue()

    if band_ndv is not None:
        ndv = band_ndv
    elif ndv is not None:
        band.SetNoDataValue(ndv)

    if ndv is not None and ndv != np.nan:
        data[data==np.nan] = ndv
    
    band.WriteArray(data)
    ds.FlushCache()


def save_r4(path, data, ndv=None):
    if ndv is not None and ndv != np.nan:
        data[data==np.nan] = ndv
    with open(path, 'wb') as outfile:
        data.flatten().astype('float32').tofile(outfile)
    ncol, nrow = data.shape[1], data.shape[0]
    with open(path + '.rsc', "w") as rsc_file:
        rsc_file.write("""\
    WIDTH                 %d
    FILE_LENGTH           %d
    XMIN                  0
    XMAX                  %d
    YMIN                  0
    YMAX                  %d""" % (ncol, nrow, ncol-1, nrow-1))


def check_dirs(path, force=False):
    for d in DIRS:
        cwd = join(path, EXPORT, d)
        if force:
            shutil.rmtree(cwd)
        Path(cwd).mkdir(parents=True, exist_ok=True)


def correct_disparity(path, target, band_disp=1, band_ndv=3, sampling=1, rm_med=True, do_mask=True, disp_path = None):
    if disp_path is None:
        disp_path = path
    data = open_gdal(path=path, band=band_disp)
    mask = open_gdal(path=disp_path, band=band_ndv)
    
    if do_mask:
        #data[mask==0] = np.nan
        np.putmask(data, mask==0, np.nan)
    if rm_med:
        data = (data - np.nanmedian(data)) * sampling
    #     median = np.nanmedian(data)
    #     data -= median
    # data *= sampling
    else:
        data *= sampling

    #save_gdal(path=target, data=data, template=path, ndv=9999)
    save_r4(path=target, data=data)


def retrieve_disparity(dir_list, working_dir, rg_sampl, az_sampl, do_mask):
    valid_pairs = []
    
    for i, d in tqdm(enumerate(dir_list)):
        # print('Start pair ({}/{}): {}'.format(i+1, len(dir_list), os.path.basename(d)))
        curr_pair = os.path.basename(d)
        disp_path = join(d, 'stereo-F.tif')
        ncc_path = join(d, 'stereo-ncc.tif')

        H_target = join(working_dir, EXPORT, 'H', 'H_{}.r4'.format(curr_pair))
        V_target = join(working_dir, EXPORT, 'V', 'V_{}.r4'.format(curr_pair))
        NCC_target = join(working_dir, EXPORT, 'NCC', 'NCC_{}.r4'.format(curr_pair))
        
        if(os.path.isfile(disp_path)):
            if os.path.isfile(H_target):
                    print('Skip {}, already exists'.format(H_target))
            else:
                correct_disparity(disp_path, H_target, band_disp=1, sampling=rg_sampl, do_mask=do_mask)
            
            if os.path.isfile(V_target):
                print('Skip {}, already exists'.format(V_target))
            else:
                correct_disparity(disp_path, V_target, band_disp=2, sampling=az_sampl, do_mask=do_mask)
            
            if os.path.isfile(NCC_target):
                print('Skip {}, already exists'.format(NCC_target))
            else:
                correct_disparity(ncc_path, NCC_target, rm_med=False, disp_path=disp_path, do_mask=do_mask)

            # print('Finished pair: {}'.format(os.path.basename(d)))
            valid_pairs.append(os.path.basename(d))
        else:
            print('No correl-F.tif file found in {}'.format(curr_pair))
            missing_correl_file = os.path.join(correl_dir, 'missing_correl_files.txt')
            mode = 'a' if os.path.isfile(missing_correl_file) else 'w'
            with open(missing_correl_file, mode) as miss_file:
                miss_file.write('{}\t{}\n'.format(curr_pair.split('_')[0], curr_pair.split('_')[1]))
    return valid_pairs


def prepare_nsbas_dir(working_dir, nsbas_process_path, orientation, pair_table, date_list_file):
    process_orient_dir = join(nsbas_process_path, orientation)
    Path(process_orient_dir).mkdir(parents=True, exist_ok=True)
    
    generate_input_inv_send(process_orient_dir)
    generate_list_pair(process_orient_dir, pair_table)
    generate_list_dates(process_orient_dir, date_list_file, pair_table)


    input_disp = join(working_dir, EXPORT, orientation)
    input_ncc = join(working_dir, EXPORT, 'NCC')
    
    ln_data_dir = join(process_orient_dir, 'LN_DATA')
    Path(ln_data_dir).mkdir(parents=True, exist_ok=True)
    
    for f in os.listdir(input_disp):
        first_dot = f.find('.')
        name, ext = f[:first_dot], f[first_dot:]
        pair = "-".join(name.split('_')[1:])
        pair_underscore = "_".join(name.split('_')[1:])
        target_disp = join(ln_data_dir, '{}{}'.format(pair, ext))
        target_ncc = join(ln_data_dir, '{}-CC{}'.format(pair, ext))
        if not os.path.islink(target_disp):
            os.symlink(join(input_disp, f), target_disp)
            print('Linked: {} to {}'.format(join(input_disp, f), target_disp))
        if not os.path.islink(target_ncc):
            ncc_file = join(input_ncc, 'NCC_' + pair_underscore + ext)
            os.symlink(ncc_file, target_ncc)
            print('Linked: {} to {}'.format(ncc_file, target_ncc))


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    work_dir = arguments['<aspsar>']
    force = arguments['--force']
    do_mask = not arguments['--no-mask']
    
    correl_dir = join(work_dir, 'STEREO')

    sampling = pd.read_csv(join(work_dir, 'sampling.txt'), sep='\t')
    az_sampl, range_sampl = sampling['AZ'][0], sampling['SR'][0]

    # prepare directories
    Path(join(work_dir, 'EXPORT')).mkdir(parents=True, exist_ok=True)

    if(force):
        print('FORCE RECOMPUTATION: REMOVE EXPORT DIRS')
    check_dirs(work_dir, force=force)

    # retrieve the dates folder YYYYMMDD_YYYYMMDD (17 chars)
    dir_list=[join(correl_dir, d) for d in os.listdir(correl_dir) if len(d) == 17]

    print('##################################')
    print('PROCESS AND COPY DISPARITY MAPS')
    print('##################################')

    pairs = retrieve_disparity(dir_list, work_dir, range_sampl, az_sampl, do_mask=do_mask)

    print(">> >> PAIRS", pairs)

    print('##################################')
    print('PREPARE NSBAS')
    print('##################################')

    nsbas_process_dir = join(work_dir, 'NSBAS_PROCESS')
    Path(nsbas_process_dir).mkdir(parents=True, exist_ok=True)
    get_date_list(pairs, nsbas_process_dir)
    
    pair_table = join(work_dir, "PAIRS", "table_pairs.txt")
    date_list_file = join(nsbas_process_dir, 'dates_list.txt')

    print('START PREPARING H DIRECTORY')
    prepare_nsbas_dir(work_dir, nsbas_process_dir, 'H', pair_table, date_list_file)
    print('FINISHED H')

    print('START PREPARING V DIRECTORY')
    prepare_nsbas_dir(work_dir, nsbas_process_dir, 'V', pair_table, date_list_file)
    print('FINISHED V')
