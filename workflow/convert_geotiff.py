#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_geotiff.py
------------
Converts the ALL2GIF results to GeoTIFF. It takes the log() of the input image.
An additional file is created (AMPLI_STACK_SIGMA_3.tif) with mean, 1/sigma and sigma as bands.

Usage: prepare_correl_dir.py --data=<path> [--f] [--s1] [--mode=<mode>]
prepare_correl_dir.py -h | --help

Options:
-h | --help         Show this screen
--data              Path to directory with linked data
--f                 Force recomputation of all files
--s1                Handle AMSTer names for Sentinel 1
--mode              Texture enhancement mode (default: log)

"""

import os
import numpy as np
from osgeo import gdal
import pandas as pd
from pathlib import Path
from math import *
import docopt
import shutil
from skimage import exposure
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

gdal.UseExceptions()

def save_to_file(data, output_path, crop=None, ndv=9999):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(output_path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    dst_band = ds.GetRasterBand(1)
    if crop is not None:
        _, xres, xrot, _, yrot, yres = ds.GetGeoTransform()
        xul = crop[0]
        yul = crop[3]
        ds.SetGeoTransform((xul, 1, xrot, yul, yrot, -1))
    dst_band.WriteArray(data)


def convert_single_file(input_file, img_dim, crop=None, mode='db', outfile=None):
    output_dir = os.path.dirname(os.path.abspath(input_file))
    geotiff_dir = os.path.join(output_dir, 'GEOTIFF')

    ncol, nrow = img_dim[0], img_dim[1]
    m = np.fromfile(input_file,dtype=np.float32)
    amp = m[:nrow*ncol].reshape((nrow,ncol))

    if crop is not None:
        # print("Crop to {}:{}[y] {}:{}[x]".format(crop[2], crop[3], crop[0], crop[1]))
        amp = amp[crop[2]:crop[3], crop[0]:crop[1]]

    if outfile is None:
        output_path_log = os.path.join(geotiff_dir, '{}_log.tif'.format(os.path.basename(input_file)))
    else:
        output_path_log = os.path.join(geotiff_dir, outfile)

    if mode == 'db':
        # if amp is amplitude and not intensity need to put at square
        amp = 10 * np.log10(np.square(amp))
    elif mode == 'rescale_intensity':
        p2, p98 = np.nanpercentile(amp, (2, 98))
        amp = exposure.rescale_intensity(amp, in_range=(p2, p98))
    elif mode == 'equalize_hist':
        amp = exposure.equalize_hist(amp)
    elif mode =='equalize_adapthist':
        data = data / np.nanmax(data)
        amp = exposure.equalize_adapthist(amp, clip_limit=0.03)
    elif mode == 'entropy':
        data = data / np.nanmax(data)
        data = entropy(data, disk(3))
    elif mode == 'log':
        amp[amp>0] = np.log(amp[amp>0])
    else:
        pass

    save_to_file(amp, output_path_log, crop=crop)
    # print("saved to", output_path_log)

    

def get_img_dimensions(input_file):
    real_path = os.path.realpath(input_file)
    i12_path = os.path.dirname(os.path.dirname(real_path))
    insar_param_file = os.path.join(i12_path, 'TextFiles', 'InSARParameters.txt')
    
    with open(insar_param_file, 'r') as f:
        lines = [''.join(l.strip().split('\t\t')[0]) for l in f.readlines()]
        jump_index = lines.index('/* -5- Interferometric products computation */')
        img_dim = lines[jump_index + 2: jump_index + 4]

    ncol, nrow = int(img_dim[0].strip()), int(img_dim[1].strip())
    return ncol, nrow, not(ncol == 0 or nrow == 0)


def get_mean_sigma_amplitude(geotiff_dir, img_dim, corrupt_file_df, crop=None):
    if crop is None:
        ncol, nrow = img_dim[0], img_dim[1]
    else:
        ncol, nrow = crop[1] - crop[0], crop[3] - crop[2]
    stack, sigma, weight = np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol))
    stack_norm, sigma_norm = np.zeros((nrow, ncol)), np.zeros((nrow, ncol))
    decorr_px = np.zeros((nrow, ncol))

    for i, f in enumerate(tqdm(os.listdir(geotiff_dir))):
        if(f in corrupt_file_df['file'].values):
            # print('Skip: {}'.format(f))    
            pass
        else:
            if('.mod_log.tif' in f):
                # print('Start: {}'.format(f))
                ds = gdal.OpenEx(os.path.join(geotiff_dir, f), allowed_drivers=['GTiff'])
                ds_band = ds.GetRasterBand(1)
                amp = ds_band.ReadAsArray()
                if np.any(np.isnan(amp)):
                    print("y'a des nannnnnnn", f)
                if np.all(np.isnan(amp)):
                    print("all NaN !", f)
                    continue
                # Convert NaNs to zero to avoid errors
                amp = np.nan_to_num(amp)
        
                # Sum amplitude
                stack += amp
                # Sum squared amplitude
                sigma += np.square(amp)
            
                stack_norm += (amp / np.nanmean(amp))
                sigma_norm += np.square((amp / np.nanmean(amp)))
                # w = np.zeros((nrow, ncol))
                # index = np.nonzero(amp)
                # w[index] = 1
                # weight += w
                weight[np.nonzero(amp)] += 1

                # decorr_px += np.isnan(amp)
                # print('Finished: {}'.format(f))
                # if i % 50 == 0:
                #     plt.figure()
                #     plt.title(i)
                #     plt.imshow(stack)
    
    # Stack is mean : divided by number of contributing pixels
    # plt.imshow(stack)
    stack[weight > 0] = stack[weight > 0] / weight[weight > 0]
    # plt.figure()
    # plt.imshow(stack)
    # plt.show()
    # Sigma is STD, variance is approximated by the difference between 
    # the mean of squared and the square of mean
    sigma[weight > 0] = np.sqrt(sigma[weight > 0] / weight[weight > 0] - (stack[weight > 0])**2)
    da = np.zeros((nrow, ncol))
    # DA amplitude dispersion = std / mean
    da[sigma > 0] = sigma[sigma > 0] / stack[sigma > 0]
    
    stack_norm[weight > 0] = stack_norm[weight > 0] / weight[weight > 0]
    sigma_norm[weight > 0] = np.sqrt(sigma_norm[weight > 0] / weight[weight > 0] - stack_norm[weight > 0]**2)

    save_to_file(stack, os.path.join(geotiff_dir, 'AMPLI_MEAN.tif'), crop=crop)
    save_to_file(sigma, os.path.join(geotiff_dir, 'AMPLI_SIGMA.tif'), crop=crop)
    save_to_file(da, os.path.join(geotiff_dir, 'AMPLI_DA.tif'), crop=crop)
    save_to_file(weight, os.path.join(geotiff_dir, 'DECORR_PIXEL.tif'), crop=crop)

    save_to_file(stack_norm, os.path.join(geotiff_dir, 'AMPLI_MEAN_NORM.tif'), crop=crop)
    save_to_file(sigma_norm, os.path.join(geotiff_dir, 'AMPLI_SIGMA_NORM.tif'), crop=crop)


def convert_all(input_path, all_file_df, geotiff_dir, s1, crop=None, mode='db'):
    for f in os.listdir(input_path):
        if(os.path.splitext(f)[1] == '.mod'):
            img_dims = get_img_dimensions(os.path.join(input_path, f))
            new_row = pd.DataFrame([{'file': f, 'ncol': img_dims[0], 'nrow': img_dims[1]}])
            all_file_df = pd.concat([all_file_df, new_row], ignore_index=True)
        
    if crop is not None:
        write_crop_to_file(os.path.dirname(geotiff_dir), crop)
    
    ncol_max = all_file_df['ncol'].value_counts().idxmax()
    nrow_max = all_file_df['nrow'].value_counts().idxmax()
    
    IMG_DIM = (int(ncol_max), int(nrow_max))
    print("X;Y: {};{}".format(IMG_DIM[0], IMG_DIM[1]))
    if crop is not None:
        print("crop x: {}:{}".format(crop[0], crop[1]))
        print("crop y: {}:{}".format(crop[2], crop[3]))
        if crop[0] > IMG_DIM[0] or crop[1] > IMG_DIM[0]:
            raise ValueError("crop exceeding x dim")
        if crop[2] > IMG_DIM[1] or crop[3] > IMG_DIM[1]:
            raise ValueError("crop exceeding x dim")
    
    ncol_differences = all_file_df.index[all_file_df['ncol'] != ncol_max]
    nrow_differences = all_file_df.index[all_file_df['nrow'] != nrow_max]
    ind_differences = ncol_differences.append(nrow_differences)
    
    corrupt_file_df = all_file_df.iloc[ind_differences]
    corrupt_file_df.to_csv(os.path.join(geotiff_dir, 'corrupt_data.txt'), sep='\t')

    print('############################')
    print('START CONVERSION')
    print('############################')

    # only process non existing files
    link_files = glob.glob(input_path + "/S1*.Z.VV.mod")
    link_files = [os.path.basename(l) for l in link_files]
    link_files.sort()

    # print("path", input_path)
    new_file_added = False
    for f in tqdm(link_files):
        outfile = os.path.join(geotiff_dir, '{}_log.tif'.format(f))
        if s1:
            # print(f)
            date = f.split("_")[2]
            outfile = date + ".VV.mod_log.tif"
        if(os.path.splitext(f)[1] == '.mod'):
            if os.path.isfile(os.path.join(geotiff_dir, outfile)):
                # print("exists")
                continue
            else:
                # if file has different extent - skip
                if(f in corrupt_file_df['file'].values):
                    print('corrupt:', f)
                    continue
                else:
                    # print('Start processing: {}'.format(f))
                    new_file_added = True
                    convert_single_file(os.path.join(input_path, f), IMG_DIM, crop=crop, mode=mode, outfile=outfile)


    # process AMPLI_STACK_SIGMA each time to always include all images
    print('Start AMPLI_MEAN and SIGMA calculation')
    if (not (os.path.isfile(os.path.join(input_path, 'GEOTIFF', "AMPLI_MEAN.tif")) or os.path.isfile(os.path.join(input_path, 'GEOTIFF_ORIGINAL', "AMPLI_MEAN.tif")))) or new_file_added:
        get_mean_sigma_amplitude(os.path.join(input_path, 'GEOTIFF'), IMG_DIM, corrupt_file_df, crop=crop)
    else:
        print("Geotiff mean already exists, use force option to recompute")

    # if s1:
    #     # from utils > rename_geotiff_S1.py
    #     geotiff_dir = os.path.join(input_path, 'GEOTIFF')
    #     geotiff_original_dir = os.path.join(input_path, 'GEOTIFF_ORIGINAL')

    #     os.rename(geotiff_dir, geotiff_original_dir)

    #     Path(geotiff_dir).mkdir(parents=True, exist_ok=True)

    #     # run through GEOTIFF_ORIGINAL - link all the data to GEOTIFF in correct format
    #     for f in os.listdir(geotiff_original_dir):
    #         if('mod_log.tif' in f):
    #             print(f.split('_')[2])
    #             target = os.path.join(geotiff_original_dir, f)#, os.path.join(geotiff_dir, '{}.VV.mod_log.tif'.format(f.split('_')[2]))
    #             if not os.path.islink(target):
    #                 os.symlink(target)


def write_crop_to_file(working_dir, crop):
    with open(os.path.join(working_dir, "crop.txt"), 'w') as cropfile:
        cropfile.write("{}\t{}\t{}\t{}".format(crop[0], crop[1], crop[2], crop[3]))


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    input_path = arguments['--data']
    all_file_df = pd.DataFrame(columns=['file', 'ncol', 'nrow'])

    force = arguments['--f']
    s1 = arguments['--s1']
    mode = arguments["--mode"]
    if mode is None:
        mode = 'db'

    geotiff_dir = os.path.join(input_path, 'GEOTIFF')

    if(force):
        shutil.rmtree(geotiff_dir)

    # create GEOTIFF directory
    Path(geotiff_dir).mkdir(parents=True, exist_ok=True)

    convert_all(input_path, all_file_df, geotiff_dir, s1, mode=mode)
