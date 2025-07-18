#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_result_export.py
-------------
Prepares an EXPORT directory to easily download the data. Adjusts the data by subtracting the median from each disparity map. Prepares the necessary files for the NSBAS processing.

Usage: prepare_result_export.py [--f] --data=<path> [--masked]
prepare_result_export.py -h | --help

Options:
-h | --help         Show this screen
--data              Path to working directory 
--f                 Force recomputation of EXPORT directory
--masked            Use the masked data as input

"""

import os, sys
import numpy as np
from osgeo import gdal
import pandas as pd
from pathlib import Path
import shutil
import docopt


def read_from_file(input_file, n_band):
    ds = gdal.OpenEx(input_file, allowed_drivers=['GTiff'])
    ds_band = ds.GetRasterBand(n_band)
    raw_disparity = ds_band.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    ncol, nrow = ds.RasterXSize, ds.RasterYSize
    # return (raw_disparity, ncol, nrow) XSize=col YSize=row
    return (raw_disparity, ncol, nrow)


def save_to_file(data, output_path, ncol, nrow):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(output_path, ncol, nrow, 1, gdal.GDT_Float32)
    dst_band = ds.GetRasterBand(1)
    dst_band.WriteArray(data)


def subtract_median(input_file, output_path, n_band, sampling):
    # img_data = (raw_disparity, ncol, nrow)
    img_data = read_from_file(input_file, n_band)
    raw_disparity, ncol, nrow = img_data[0], img_data[1], img_data[2]
    median = np.nanmedian(raw_disparity)

    adj_disparity = np.zeros((nrow, ncol))
    adj_disparity = (raw_disparity - median) * sampling
    # add conversion from pixel to m displacement here
    save_to_file(adj_disparity, output_path, ncol, nrow)


def get_cc_map(input_file, output_path, n_band):
    # img_data = (correlation coefficient, ncol, nrow)
    img_data = read_from_file(input_file, n_band)
    cc, ncol, nrow = img_data[0], img_data[1], img_data[2]

    save_to_file(cc, output_path, ncol, nrow)


def generate_input_inv_send(out_dir):
        # CHECK INPUT VARIABLE AT END OF STRING -> SET ALL TO 0
    f = open(os.path.join(out_dir, "input_inv_send"), "w")
    # set starting parameters as suggested in documentation
    f.write("""\
0.003  #  temporal smoothing weight, gamma liss **2 (if <0.0001, no smoothing)
1     #   mask pixels with large RMS misclosure  (y=0;n=1)
1.5    #  threshold for the mask on RMS misclosure (in same unit as input files)
1      #  range and azimuth downsampling (every n pixel)
0      #  iterations to correct unwrapping errors (y:nb_of_iterations,n:0)
3      #  iterations to weight pixels of interferograms with large residual? (y:nb_of_iterations,n:0)
0.5    #  Scaling value for weighting residuals (1/(res**2+value**2)) (in same unit as input files) (Must be approximately equal to standard deviation on measurement noise)
2      #  iterations to mask (tiny weight) pixels of interferograms with large residual? (y:nb_of_iterations,n:0)
4.     #  threshold on residual, defining clearly wrong values (in same unit as input files)
1      #  outliers elimination by the median (only if nsamp>1) ? (y=0,n=1)
list_dates
0      #  sort by date (0) ou by another variable (1) ?
list_pair
%d     #  interferogram format (RMG : 0; R4 :1) (date1-date2_pre_inv.unw or date1-date2.r4)
3100.  #  include interferograms with bperp lower than maximal baseline
%d      #  Weight input interferograms by coherence or correlation maps ? (y:0,n:1)
%d      #  coherence file format (RMG : 0; R4 :1) (date1-date2.cor or date1-date2-CC.r4)
1      #  minimal number of interferams using each image
1      #  interferograms weighting so that the weight per image is the same (y=0;n=1)
0.5    #  maximum fraction of discarded interferograms
0      #  Would you like to restrict the area of inversion ?(y=1,n=0)
1 735 1500 1585  #  Give four corners, lower, left, top, right in file pixel coord
1      #  referencing of interferograms by bands (1) or corners (2) ? (More or less obsolete)
5      #  band NW -SW(1), band SW- SE (2), band NW-NE (3), or average of three bands (4) or no referencement (5) ?
1      #  Weigthing by image quality (y:0,n:1) ? (then read quality in the list of input images)
%d     #  Weigthing by interferogram variance (y:0,n:1) or user given weight (2)?
1      #  use of covariance (y:0,n:1) ? (Obsolete)
1      #  Adjust functions to phase history ? (y:1;n:0) Require to use smoothing option (smoothing coefficient) !
0      #  compute DEM error proportional to perpendicular baseline ? (y:1;n:0)
0 2003.0     #  include a step function ? (y:1;n:0)
0      #  include a cosinus / sinus function ? (y:1;n:0)
1      #  smoothing by Laplacian, computed with a scheme at 3pts (0) or 5pts (1) ?
2      #  weigthed smoothing by the average time step (y :0 ; n : 1, int : 2) ?
1      # put the first derivative to zero (y :0 ; n : 1)?
    """ % (1, 1, 1, 1))
    f.close()
    

def process_single_disparity_NSBAS(input_file, direction, masked):
    """
    input_file = either to H_wm or V_wm file

    direction - information of direction; either H or V (for naming of resulting files)
    """
    curr_pair = os.path.basename(input_file).split('-')[0]
    dates_pair = '{}-{}'.format(curr_pair.split('_')[0], curr_pair.split('_')[1])

    # make distinction if normal input files or masked input files should be used
    if(masked):
        if(direction == 'H'):
            out_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)), 'NSBAS', 'MASKED', 'H')
            #n_band = 1
        else:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)), 'NSBAS', 'MASKED', 'V')
    else:
        # set n_band variable based on direciton info
        # H = 1; V = 2
        if(direction == 'H'):
            out_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)), 'NSBAS', 'H')
            #n_band = 1
        else:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)), 'NSBAS', 'V')
            #n_band = 2

    out_file = os.path.join(out_dir, '{}_{}.r4'.format(dates_pair, direction))

    # just read first band because it reads the processed files for each direction (either H or V)
    img_data = read_from_file(input_file, 1)
    los, ncol, nrow = img_data[0], img_data[1], img_data[2]

    fid = open(out_file, 'wb')
    los.flatten().astype('float32').tofile(fid)
    fid.close()

    out_rsc = os.path.join(out_dir, '{}_{}.r4.rsc'.format(dates_pair, direction))
    f = open(out_rsc, 'w')
    f = open(out_rsc, "w")
    f.write("""\
      WIDTH                 %d
      FILE_LENGTH           %d
      XMIN                  0
      XMAX                  %d
      YMIN                  0
      YMAX                  %d
      """ % (ncol, nrow, ncol-1, nrow-1))
    f.close()


def get_date_list(pair_list, out_dir):
    """creates txt file in work_dir/NSBAS/date_list.txt"""
    dates = []
    for p in pair_list:
        #print(str(p))
        dates.append(p.split('_')[0])
        dates.append(p.split('_')[1])

    dates = list(set(dates))

    with open(os.path.join(out_dir, 'dates_list.txt'), 'w') as f:
        for d in sorted(dates):
            f.write('{}\n'.format(d))


def prepare_NSBAS(data_dir, masked):
    # to only get unique values (bc for each pair 2 files(V&H)) - transform to set
    pair_set = set([d.split('-')[0] for d in os.listdir(data_dir)])
    # convert to list to keep list data type
    pair_list = list(pair_set)

    for i, f in enumerate(pair_list):
        print('Start pair ({}/{}): {}'.format(i+1, len(pair_list), f))
        if(masked):
            h_file = os.path.join(data_dir, '{}-F-H_wm_MASK.tif'.format(f))
            v_file = os.path.join(data_dir, '{}-F-V_wm_MASK.tif'.format(f))
        else:
            h_file = os.path.join(data_dir, '{}-F-H_wm.tif'.format(f))
            v_file = os.path.join(data_dir, '{}-F-V_wm.tif'.format(f))

        print('Start: {}'.format(os.path.basename(h_file)))
        process_single_disparity_NSBAS(h_file, 'H', masked)
        print('Finished: {}'.format(os.path.basename(h_file)))

        print('Start: {}'.format(os.path.basename(v_file)))
        process_single_disparity_NSBAS(v_file, 'V', masked)
        print('Finished: {}'.format(os.path.basename(v_file)))

    out_dir = os.path.join(os.path.dirname(data_dir), 'NSBAS')

    generate_input_inv_send(out_dir)

    get_date_list(pair_list, out_dir)
    

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    # work_dir is data_dir from before, because everything is in one directory
    work_dir = arguments['--data']

    # check for force recomputation - will remove current EXPORT/ADJUSTED, /NSBAS, /RAW directory 
    force = arguments['--f']

    # check if masked option is set - will use masked files as input
    masked = arguments['--masked']

    # correl_dir is like working_dir, all the processing results are stored in data_dir/CORREL
    correl_dir = os.path.join(work_dir, 'STEREO')

    # get azimuth and slant range sampling for pixel to m conversion
    sampling_file = os.path.join(work_dir, 'sampling.txt')
    sampling = pd.read_csv(sampling_file, sep='\t')
    az_sampl, range_sampl = sampling['AZ'][0], sampling['SR'][0]

    # prepare directories
    exp_dir = os.path.join(work_dir, 'EXPORT')
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    raw_dir = os.path.join(exp_dir, 'RAW')
    adj_dir = os.path.join(exp_dir, 'ADJUSTED')
    nsbas_dir = os.path.join(exp_dir, 'NSBAS')

    cc_dir = os.path.join(exp_dir, 'CC')

    if(force):
        print('FORCE RECOMPUTATION: REMOVE EXPORT/ADJUSTED, EXPORT/NSBAS, EXPORT/RAW, EXPORT/CC')
        shutil.rmtree(raw_dir)
        shutil.rmtree(adj_dir)
        shutil.rmtree(nsbas_dir)
        shutil.rmtree(cc_dir)
        
    # check and create subdirectories
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(adj_dir).mkdir(parents=True, exist_ok=True)

    Path(nsbas_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(nsbas_dir, 'H')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(nsbas_dir, 'V')).mkdir(parents=True, exist_ok=True)

    # prepare the masked results in NSBAS/MASKED/H||V
    if(masked):
        masked_nsbas_dir = os.path.join(nsbas_dir, 'MASKED')
        Path(masked_nsbas_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(masked_nsbas_dir, 'H')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(masked_nsbas_dir, 'V')).mkdir(parents=True, exist_ok=True)


    Path(cc_dir).mkdir(parents=True, exist_ok=True)

    if(masked):
        masked_input_dir = os.path.join(exp_dir, 'MASKED')
        if(not os.path.exists(masked_input_dir) or len(os.listdir(masked_input_dir)) <= 2):
            print('EXPORT/MASKED is not existing/empty, run utils/mask_correl_results_cc.py')
        else:
            print('EXPORT/MASKED exists, can start processing')

    # get list of only DATE1_DATE2 directories
    dir_list=[os.path.join(correl_dir, d) for d in os.listdir(correl_dir) if len(d) == 17]

    print('##################################')
    print('PROCESS AND COPY DISPARITY MAPS')
    print('##################################')

    for i, d in enumerate(dir_list):
        print('Start pair ({}/{}): {}'.format(i+1, len(dir_list), os.path.basename(d)))
        curr_pair = os.path.basename(d)
        # use here the complete correl-F instead of the disparitydebug results
        v_path = os.path.join(d, 'stereo-F.tif')
        h_path = os.path.join(d, 'stereo-F.tif')
        cc_path = os.path.join(d, 'stereo-ncc.tif')
        
        if(os.path.isfile(v_path)):
            # add n_band based on direction

            if(os.path.isfile(os.path.join(adj_dir, '{}-F-H_wm.tif'.format(curr_pair)))):
                    print('Skip {}-F-H_wm.tif, already exists'.format(curr_pair))
            else:
                subtract_median(h_path, os.path.join(adj_dir, '{}-F-H_wm.tif'.format(curr_pair)), 1, range_sampl)
            
            if(os.path.isfile(os.path.join(adj_dir, '{}-F-V_wm.tif'.format(curr_pair)))):
                print('Skip {}-F-V_wm.tif, already exists'.format(curr_pair))
            else:
                subtract_median(v_path, os.path.join(adj_dir, '{}-F-V_wm.tif'.format(curr_pair)), 2, az_sampl)
            
            if(os.path.isfile(os.path.join(adj_dir, '{}_NCC.tif'.format(curr_pair)))):
                print('Skip {}_NCC.tif, already exists'.format(curr_pair))
            else:
                # gets the 3rd band from correl-F.tif which contains correlation coeffiecent map
                get_cc_map(cc_path, os.path.join(cc_dir, '{}_NCC.tif'.format(curr_pair)), 1)

            # instead of copying, just link raw data to save space (maybe remove later)
            #shutil.copy(h_path, os.path.join(raw_dir, '{}-F-H.tif'.format(curr_pair)))
            #shutil.copy(v_path, os.path.join(raw_dir, '{}-F-V.tif'.format(curr_pair)))

            if(os.path.isfile(os.path.join(raw_dir, '{}-F-H.tif'.format(curr_pair)))):
                print('Skip linking of {}'.format(curr_pair))
            else:
                os.symlink(h_path, os.path.join(raw_dir, '{}-F-H.tif'.format(curr_pair)))
                os.symlink(v_path, os.path.join(raw_dir, '{}-F-V.tif'.format(curr_pair)))

            print('Finished pair: {}'.format(os.path.basename(d)))
        else:
            print('No correl-F.tif file found in {}'.format(curr_pair))
            # add to print names to list
            missing_correl_file = os.path.join(correl_dir, 'missing_correl_files.txt')
            if(os.path.isfile(missing_correl_file)):
                with open(missing_correl_file, 'a') as miss_file:
                    miss_file.write('{}\t{}\n'.format(curr_pair.split('_')[0], curr_pair.split('_')[1]))
            else:    
                with open(missing_correl_file, 'w') as miss_file:
                    miss_file.write('{}\t{}\n'.format(curr_pair.split('_')[0], curr_pair.split('_')[1]))

    # maybe add here masking based on CC_map - set input option to apply  optional 

    print('##################################')
    print('PREPARE NSBAS')
    print('##################################')

    #TODO: add update option -> import for the links/textfiles -> f.e. watch out, don't overwrite input_inv_send

    if(masked):
        prepare_NSBAS(masked_input_dir, masked)
    else:
        prepare_NSBAS(adj_dir, masked)
