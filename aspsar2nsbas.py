#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aspsar2nsbas.py
--------------
Prepare the directory structure for further processing. Link all ALL2GIF results in the given destination dir.

Usage: aspsar2nsbas.py <aspsar> [--force] [--cc-mask] [--da-mask=<da-mask>] [--stereo=<stereo>] [--ref-area=<ref-area>] [--debug | -d] [-v | --verbose]
aspsar2nsbas.py -h | --help

Options:
    -h | --help         Show this screen
    --aspsar              Path to aspsar processing directory
    --s1
    -d --debug
    -v --verbose

"""

import docopt
import logging
import os
from os.path import join
from pathlib import Path
import shutil
from tqdm import tqdm
import numba
from osgeo import gdal
from time import time
import numpy as np
import glob
# import pandas as pd

import scipy.optimize as opt
import scipy.linalg as lst

from workflow.prepare_result_export import generate_input_inv_send, get_date_list
from workflow.prepare_nsbas_process import generate_list_dates, generate_list_pair

gdal.UseExceptions()
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

DIRS = ['H', 'V', 'NCC']
EXPORT = 'EXPORT'


import os, sys, subprocess

def sh(cmd: str, shell: bool = True):
    return subprocess.run(
        cmd,
        shell=shell,
        env=os.environ,
    )

def open_gdal(path, band=1):
    logger.info("opening file {}".format(path))
    t0 = time()
    ds = gdal.Open(path)
    t1 = time()
    bd = ds.GetRasterBand(band)
    t2 = time()
    ndv = bd.GetNoDataValue()
    ## Very slow
    data = bd.ReadAsArray()
    ##
    t3 = time()

    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    t4 = time()
    logger.debug("{}-{}-{}-{}".format(t1-t0, t2-t1, t3-t2, t4-t3))
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
    logger.info("Saved gdal {}".format(path))


def save_r4(path, data, ndv=None):
    if ndv is not None and ndv != np.nan:
        data[data==np.nan] = ndv
    with open(path, 'wb') as outfile:
        data.flatten().astype('float32').tofile(outfile)
    ncol, nrow = data.shape[1], data.shape[0]
    generate_rsc(path, ncol, nrow)
    generate_hdr(path, ncol, nrow)
    logger.info("Saved REAL4 {}".format(path))
        

def generate_rsc(target, ncol, nrow):
    with open(target + '.rsc', "w") as rsc_file:
        rsc_file.write("""\
    WIDTH                 %d
    FILE_LENGTH           %d
    XMIN                  0
    XMAX                  %d
    YMIN                  0
    YMAX                  %d""" % (ncol, nrow, ncol-1, nrow-1))


def generate_hdr(target, ncol, nrow, nband=1):
    with open(os.path.splitext(target)[0] + '.hdr', "w") as hdr_file:
        hdr_file.write("""\
ENVI
samples = {}
lines   = {}
bands   = {}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = 0
band names = {
Band 1}
data ignore value = 3.4028234663852886e+38
default bands = {
1}""".format(ncol, nrow, nband))


def check_dirs(path, force=False):
    for d in DIRS:
        cwd = join(path, EXPORT, d)
        if force:
            shutil.rmtree(cwd)
        Path(cwd).mkdir(parents=True, exist_ok=True)


def deramp(data):
    """ Derived from clean_raster, Pygdalsar, Simon Daout
    """
    # print(data)
    data_max, data_min = np.nanpercentile(data, 99.5), np.nanpercentile(data, 0.5)
    # print("min max", data_min, data_max)
    # print("cond", np.logical_and(data < data_min, data > data_max))
    to_clean = np.nonzero(np.logical_or(data < data_min, data > data_max))
    # print("indices to clean", to_clean)
    data_clean = data.copy()
    # print(data_clean)
    data_clean[to_clean] = np.nan
    # print(data_clean)

    # Fetch index
    index = np.nonzero(~np.isnan(data_clean))
    # print("index:",index)
    mi = data[index].flatten()
    az = np.asarray(index[0])
    rg = np.asarray(index[1])

    G=np.zeros((len(mi),3))
    G[:,0] = rg
    G[:,1] = az
    G[:,2] = 1

    x0 = lst.lstsq(G,mi)[0]
    _func = lambda x: np.sum(((np.dot(G,x)-mi))**2)
    _fprime = lambda x: 2*np.dot(G.T, (np.dot(G,x)-mi))
    pars = opt.fmin_slsqp(_func,x0,fprime=_fprime,iter=2000,full_output=True,iprint=0)[0]
    a, b, c = pars[0], pars[1], pars[2]
    logger.info('Remove ramp %f x  + %f y + %f'%(a,b,c))

    G = np.zeros((len(data.flatten()), 3))
    ncols = data.shape[1]
    nlines = data.shape[0]
    for i in range(nlines):
        G[i*ncols:(i+1)*ncols,0] = np.arange((ncols))
        G[i*ncols:(i+1)*ncols,1] = i
    G[:,2] = 1
    # apply ramp correction
    ramp = np.dot(G,pars)
    ramp = ramp.reshape(nlines, ncols)
    # plt.figure()
    # plt.imshow(data, cmap="RdBu_r")
    # plt.title("data")

    # plt.figure()
    # plt.imshow(ramp, cmap="RdBu_r")
    # plt.title("ramp")

    # plt.figure()
    # plt.imshow(data - ramp, cmap="RdBu_r")
    # plt.title("data deramp")
    # plt.show()
    return data - ramp


def correct_disparity_sh(folder, path, target, band_disp=1, band_ndv=3, sampling=1, disp_path = None, do_deramp=True):
    if disp_path is None:
        disp_path = path
    
    ds = gdal.Open(disp_path)
    nrow, ncol = ds.RasterXSize, ds.RasterYSize
    del ds

    intermediary_files = []

    # (1) Isolate and apply sampling
    cmd1 = '''gdal_calc -A {} --A_band={} -B {} --B_band={} --calc="numpy.where(B>=1, A * {}, numpy.nan)" --outfile={}'''.format(
        path,
        band_disp,
        disp_path,
        band_ndv,
        sampling,
        target + "_meters.tif"
    )
    logger.debug(cmd1)
    sh(cmd1)
    current_file = target + "_meters.tif"
    intermediary_files.append(current_file)

    if do_deramp:
        # (2) Remove residual ramp
        # cmd2 = '''remove_quad_ramp.py {} {}'''.format(
        #     current_file,
        #     target + "_deramp.tif"
        # )
        dem = os.path.join(folder, "radar_dem.tif")
        aspect = os.path.join(folder, "dem_east_aspect_rads.tif")
        if os.path.isfile(aspect):
            logger.debug("remove aspect ramp")
            cmd2 = '''deramp_ransac.py {} --outfile={} --chunk-size=4096,4096 --add-data={} --cyclic-data --save-coeffs --ramp=linear'''.format(
                current_file,
                target + "_deramp.tif",
                aspect
            )
        else:
            logger.debug("remove ramp without topo")
            cmd2 = '''remove_topo_ramp.py {} {} --chunk-size=4096,4096'''.format(
                current_file,
                target + "_deramp.tif"
            )
        logger.debug(cmd2)
        sh(cmd2)
        current_file = target + "_deramp.tif"
        intermediary_files.append(current_file)

    # (3) Reference to median
    cmd3 = '''gdal_calc -A {} --calc="A - numpy.nanmedian(A)" --outfile={}'''.format(
        current_file,
        target + "_referenced.tif"
    )
    current_file = target + "_referenced.tif"
    intermediary_files.append(current_file)
    print(cmd3)
    sh(cmd3)

    # (4) Translate to ENVI
    cmd4 = '''gdal_translate {} {} -of ENVI'''.format(
        current_file,
        target
    )
    logger.debug(cmd4)
    sh(cmd4)

    # (5) Generate .rsc header
    logger.debug("gen header")
    generate_rsc(target, nrow, ncol)

    # (6) Remove intermediate files
    logger.debug("rm files")
    for f in intermediary_files:
        os.remove(f)


def correct_disparity(path, target, band_disp=1, band_ndv=3, sampling=1, rm_med=True, cc_mask=True, da_mask=None, disp_path = None, do_deramp=True, ref_area=None):
    if disp_path is None:
        disp_path = path

    # # (1) Isolate and apply sampling
    # cmd1 = '''gdal_calc -A {} --A_band={} -B {} --B_band={} --calc="numpy.where(B>=1, A * {}, numpy.nan)" --outfile={}'''.format(
    #     path,
    #     band_disp,
    #     disp_path,
    #     band_ndv,
    #     sampling,
    #     target + "_meters.tif"
    # )
    # print(cmd1)
    # sh(cmd1)

    # # (2) Remove residual ramp
    # cmd2 = '''remove_quad_ramp.py {} {}'''.format(
    #     target + "_meters.tif",
    #     target + "_deramp.tif"
    # )
    # print(cmd2)
    # sh(cmd2)

    # # if rm_med:
    # #     # (3) Reference to median
    # #     cmd3 = '''gdal_calc -A {} --calc="A - numpy.nanmed(A)" --outfile={}'''.format(
    # #         target + "_deramp.tif",
    # #         target + "_referenced.tif"
    # #     )
    # #     print(cmd3)
    # #     sh(cmd3)

    # # (4) Translate to ENVI
    # cmd4 = '''gdal_translate {} {} -of ENVI'''.format(
    #     target + "_deramp.tif",
    #     target
    # )
    # print(cmd4)
    # sh(cmd4)

    # # (5) Generate .rsc header
    # print("gen header")
    # generate_rsc(target)

    # # (6) Remove intermediate files
    # print("rm files")
    # # os.remove(target + "_meters.tif")
    # # os.remove(target + "_deramp.tif")
    # # os.remove(target + "_referenced.tif")

    logger.info("correct disparity {}".format(path))
    t0 = time()
    if disp_path is None:
        disp_path = path
    logger.info("open files")
    data = open_gdal(path=path, band=band_disp)
    mask = open_gdal(path=disp_path, band=band_ndv)
    # data = data[:1000, :1000]
    # mask = mask[:1000, :1000]
    t1=time()
    print(t1-t0)
    
    if cc_mask:
        logger.info("use ndv band")
        #data[mask==0] = np.nan
        np.putmask(data, mask==0, np.nan)
    if da_mask is not None:
        # data, threshold
        logger.info("use da masking")
        da_data, da_threshold = da_mask
        np.putmask(data, da_data > da_threshold, np.nan)
    if do_deramp:
        logger.info("deramp")
        deramp(data)
    if ref_area is not None:
        data[~np.isnan(data)] = correct_median_reference(data[~np.isnan(data)], sampling, ref_area)
    # if rm_med:
    #     # median is already to zero with the ramp
    #     pass

    #     # logger.info("reference to median {}".format(np.nanmedian(data)))
    #     # data[~np.isnan(data)] = correct_median(data[~np.isnan(data)], sampling)


    #     # if ref_area is None:
    #     #     data[~np.isnan(data)] = correct_median(data[~np.isnan(data)], sampling)
    #     # else:
    #     #     data[~np.isnan(data)] = correct_median_reference(data[~np.isnan(data)], sampling, ref_area)
    # #     median = np.nanmedian(data)
    # #     data -= median
    # # data *= sampling
    else:
        logger.info("apply sampling")
        data *= sampling
    t2=time()
    #save_gdal(path=target, data=data, template=path, ndv=9999)
    logger.info("saving to r4")
    save_r4(path=target, data=data)
    t3=time()
    print("time (loading, corrections, saving)", t1-t0, t2-t1, t3-t2)


@numba.jit(nopython=True)
def correct_median(array, coeff):
    return (array - np.median(array)) * coeff


@numba.jit(nopython=True)
def correct_median_reference(array, coeff, crop):
    return (array - np.median(array[crop[2]:crop[3], crop[1]:crop[0]])) * coeff


def retrieve_disparity(dir_list, working_dir, rg_sampl, az_sampl, cc_mask, da_mask, da_file=None, ref_area=None):
    valid_pairs = []
    if da_mask is not None:
        da = open_gdal(da_file)
        da_mask = da, da_mask
    
    for i, d in enumerate(tqdm(dir_list)):
        # print('Start pair ({}/{}): {}'.format(i+1, len(dir_list), os.path.basename(d)))
        logger.debug("Starting pair {}: {}".format(i + 1, os.path.basename(d)))
        curr_pair = os.path.basename(d)
        # find filtered and NCC results
        disp_path = join(d, 'stereo-F.tif')
        ncc_path = join(d, 'stereo-ncc.tif')

        # Define the three export files
        H_target = join(working_dir, EXPORT, 'H', 'H_{}.r4'.format(curr_pair))
        V_target = join(working_dir, EXPORT, 'V', 'V_{}.r4'.format(curr_pair))
        NCC_target = join(working_dir, EXPORT, 'NCC', 'NCC_{}.r4'.format(curr_pair))
        
        if(os.path.isfile(disp_path)):
            if os.path.isfile(H_target):
                    logger.info('Skip {}, already exists'.format(H_target))
            else:
                logger.debug("Correct disparity H")
                correct_disparity_sh(working_dir, disp_path, H_target, sampling=rg_sampl, band_disp=1)
                # correct_disparity(disp_path, H_target, band_disp=1, sampling=rg_sampl, cc_mask=cc_mask, da_mask=da_mask, ref_area=ref_area)
            
            if os.path.isfile(V_target):
                logger.info('Skip {}, already exists'.format(V_target))
            else:
                logger.debug("Correct disparity V")
                correct_disparity_sh(working_dir, disp_path, V_target, sampling=az_sampl, band_disp=2)
                # correct_disparity(disp_path, H_target, band_disp=2, sampling=az_sampl, cc_mask=cc_mask, da_mask=da_mask, ref_area=ref_area)
            
            if os.path.isfile(NCC_target):
                logger.info('Skip {}, already exists'.format(NCC_target))
            else:
                logger.debug("Correct disparity NCC")
                correct_disparity_sh(working_dir, ncc_path, NCC_target, do_deramp=False, disp_path=disp_path)
                # correct_disparity(ncc_path, NCC_target, rm_med=False, disp_path=disp_path, cc_mask=cc_mask, da_mask=da_mask, do_deramp=False)

            # print('Finished pair: {}'.format(os.path.basename(d)))
            valid_pairs.append(os.path.basename(d))
        else:
            logger.info('No correl-F.tif file found in {}'.format(curr_pair))
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
        name, ext = os.path.splitext(f)
        if ext == ".r4" or ext == ".rsc":
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
    # a = np.array([[4,2,3,5,6], [4, 5, 6, 9, 7], [4,5,6,7,8], [0,1,5,6,8]], dtype=float)
    # print(deramp(a))
    # quit()

    arguments = docopt.docopt(__doc__)

    if arguments["--verbose"]:
        if arguments["--debug"]:
            logging.basicConfig(
                level=logging.DEBUG, format="%(levelname)s: %(asctime)s | %(message)s"
            )
        else:
            logging.basicConfig(
                level=logging.INFO, format="%(levelname)s: %(asctime)s | %(message)s"
            )
    else:
        logging.basicConfig(
            level=logging.WARNING, format="%(levelname)s: %(asctime)s | %(message)s"
        )
    logger.debug("CLI arguments: {}".format(arguments))

    work_dir = arguments['<aspsar>']
    force = arguments['--force']
    cc_mask = arguments['--cc-mask']
    da_mask = None if arguments['--da-mask'] is None else float(arguments['--da-mask'])
    
    if da_mask is None:
        da_file = None
    else:
        da_file = join(work_dir, 'GEOTIFF', 'AMPLI_dSIMGA.tif')
        if not os.path.isfile(da_file):
            da_file = join(work_dir, 'GEOTIFF_ORIGINAL', 'AMPLI_dSIMGA.tif')
        if not os.path.isfile(da_file):
            da_file = join(work_dir, 'GEOTIFF', 'AMPLI_DA.tif')
            if not os.path.isfile(da_file):
                raise FileNotFoundError("DA file (AMPLI_DA.tif) not found")
    
    correl_dir = arguments["--stereo"] if arguments["--stereo"] is not None else join(work_dir, 'STEREO')
    ref_area = arguments["--ref-area"]
    if ref_area is not None:
        ref_area = [int(r) for r in ref_area.split(',', 4)]

    # read az and rg pixel resolution
    # sampling = pd.read_csv(join(work_dir, 'sampling.txt'), sep='\t')
    # az_sampl, range_sampl = sampling['AZ'][0], sampling['SR'][0]
    az_sampl, range_sampl = np.loadtxt(join(work_dir, 'sampling.txt'), unpack=True, skiprows=1)

    # prepare directories
    Path(join(work_dir, 'EXPORT')).mkdir(parents=True, exist_ok=True)

    if(force):
        logger.info('FORCE RECOMPUTATION: REMOVE EXPORT DIRS')
    check_dirs(work_dir, force=force)

    # retrieve the dates folder YYYYMMDD_YYYYMMDD (17 chars)
    # dir_list = [join(correl_dir, d) for d in os.listdir(correl_dir) if len(d) == 17]
    dir_list = [join(correl_dir, d) for d in os.listdir(correl_dir)]
    dir_list.sort()

    logger.info('##################################')
    logger.info('PROCESS AND COPY DISPARITY MAPS')
    logger.info('##################################')

    pairs = retrieve_disparity(dir_list, work_dir, range_sampl, az_sampl, cc_mask=cc_mask, da_mask=da_mask, da_file=da_file, ref_area=ref_area)

    logger.info(">> >> PAIRS", pairs)

    logger.info('##################################')
    logger.info('PREPARE NSBAS')
    logger.info('##################################')

    nsbas_process_dir = join(work_dir, 'NSBAS_PROCESS')
    Path(nsbas_process_dir).mkdir(parents=True, exist_ok=True)
    get_date_list(pairs, nsbas_process_dir)
    
    pair_table = join(work_dir, "PAIRS", "table_pairs.txt")
    date_list_file = join(nsbas_process_dir, 'dates_list.txt')

    logger.info('START PREPARING H DIRECTORY')
    prepare_nsbas_dir(work_dir, nsbas_process_dir, 'H', pair_table, date_list_file)
    logger.info('FINISHED H')

    logger.info('START PREPARING V DIRECTORY')
    prepare_nsbas_dir(work_dir, nsbas_process_dir, 'V', pair_table, date_list_file)
    logger.info('FINISHED V')
