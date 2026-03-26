#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aspsar2export.py
-----------
Prepare an EXPORT directory given the results of ASPSAR correlation (STEREO)

Usage: aspsar2export.py <stereo> <export> [--pairs=<pairs>] [--ramp=<ramp>] [--verbose | -v] [--tr=<tr>] [--force]
aspsar2export.py -h | --help

Options:
  -h, --help       Show this screen
  -v, --verbose
"""

import os
import docopt
import numpy as np
import subprocess
import logging
from osgeo import gdal

logger = logging.getLogger(__name__)


def sh(cmd: str, shell: bool = True):
    return subprocess.run(
        cmd,
        shell=shell,
        env=os.environ,
    )


class Pair:
    def __init__(self, date1, date2):
        self.date1 = date1
        self.date2 = date2
        self.stereo_result = None
        self.stereo_ncc = None
        self.local_result = None
        self.local_ncc = None
    
    def __str__(self):
        return "{}_{}".format(self.date1, self.date2)

    @staticmethod
    def read_from_file(table) -> list:
        pairs = np.loadtxt(table, usecols=(0, 1), skiprows=2, dtype=str)
        pairs = [Pair(p[0], p[1]) for p in pairs]
        return pairs

    def search_stereo(self, stereo_dir, missing_f, missing_ncc):
        target = os.path.join(stereo_dir, "{}_{}".format(self.date1, self.date2), "stereo-F.tif")
        if os.path.isfile(target):
            self.stereo_result = target
        else:
            logger.info("File not found: {}".format(target))
            missing_f.append(str(self))

        target = os.path.join(stereo_dir, "{}_{}".format(self.date1, self.date2), "stereo-ncc.tif")
        if os.path.isfile(target):
            self.stereo_ncc = target
        else:
            logger.info("File not found: {}".format(target))
            missing_ncc.append(str(self))

        return self

    def apply_corrections(self, export_dir, stereo_res, force):
        for i, pref in enumerate(["H", "V"]):
            target = os.path.join(export_dir, pref, pref + "_" + str(self))
            if not os.path.isfile(target + "_meters.tif") or force:
                # apply sampling and ndv
                cmd = '''gdal_calc -A {} --A_band={} -B {} --B_band={} --calc="numpy.where(B>=1, A * {}, numpy.nan)" --outfile={}'''.format(
                    self.local_result,
                    1 if pref == "H" else 2,
                    self.local_result,
                    3,
                    stereo_res[i],
                    target + "_meters.tif"
                )
                logger.info(cmd)
                sh(cmd)
            else:
                logger.info("Corrected raster {} already exists. Keeping.".format(target))

    def remove_median(self, export_dir):
        for pref in ["H", "V"]:
            target = os.path.join(export_dir, pref, pref + "_" + str(self) + "_meters.tif_deramp.tif")
            if not os.path.isfile(target + "_median.tif") or force:
                cmd = '''gdal_calc -A {} --calc="A - numpy.nanmedian(A)" --outfile={}'''.format(
                    target,
                    target + "_median.tif"
                )
                logger.info(cmd)
                sh(cmd)
            else:
                logger.info("Median removed raster {} already exists. Keeping.".format(target))

    def switch_to_envi(self, export_dir):
        for pref in ["H", "V", "NCC"]:
            target = os.path.join(export_dir, pref, pref + "_" + str(self) + "_meters.tif_deramp.tif_median.tif")
            if pref == "NCC":
                target = os.path.join(export_dir, pref, pref + "_" + str(self) + ".tif")
            output = os.path.join(export_dir, pref, pref + "_" + str(self) + ".r4")
            if not os.path.isfile(output) or force:
                cmd = '''gdal_translate {} {} -of ENVI -co INTERLEAVE=BSQ'''.format(
                    target,
                    output
                )
                logger.info(cmd)
                sh(cmd)
                generate_rsc(output)
            else:
                logger.info("ENVI raster {} already exists. Keeping.".format(target))

    def clean_files(self, export_dir):
        for pref in ["H", "V"]:
            target = os.path.join(export_dir, target, pref + "_" + str(self))
            for f in [
                target + "_meters.tif",
                target + "_meters.tif_deramp.tif",
                target + "_meters.tif_deramp.tif_median.tif",
            ]:
                if os.path.isfile(f):
                    os.remove(f)

    def link_or_resample(self, export_dir, res=None):
        # stereo-F.tif
        target = os.path.join(export_dir, str(self) + ".tif")
        if res is not None and (force or not os.path.isfile(target)):
            cmd = "gdal_translate {} {} -tr {} {}".format(
                self.stereo_result, target, str(int(res)), str(int(res))
            )
            logger.info(cmd)
            sh(cmd)
        elif res is None and not os.path.islink(target):
            logger.info("Linking {} to {}".format(self.stereo_result, target))
            os.symlink(os.path.abspath(self.stereo_result), target)
        else:
            logger.info("Target {} exists. Keeping.".format(target))
        self.local_result = target

        # stereo-ncc.tif
        target = os.path.join(export_dir, 'NCC', 'NCC_' + str(self) + ".tif")
        if res is not None and (force or not os.path.isfile(target)):
            cmd = "gdal_translate {} {} -tr {} {}".format(
                self.stereo_ncc, target, str(int(res)), str(int(res))
            )
            logger.info(cmd)
            sh(cmd)
        elif res is None and not os.path.islink(target):
            logger.info("Linking {} to {}".format(self.stereo_result, target))
            os.symlink(os.path.abspath(self.stereo_ncc), target)
        else:
            logger.info("Target {} exists. Keeping.".format(target))
        self.local_ncc = target

        return self


def generate_rsc(gdal_raster):
    ds = gdal.Open(gdal_raster)
    ncol, nrow = ds.RasterXSize, ds.RasterYSize
    ds = None
    with open(gdal_raster + '.rsc', "w") as rsc_file:
        rsc_file.write("""\
    WIDTH                 %d
    FILE_LENGTH           %d
    XMIN                  0
    XMAX                  %d
    YMIN                  0
    YMAX                  %d""" % (ncol, nrow, ncol-1, nrow-1))


def apply_ramps(export_dir, ramp_opts, force=False):
    add = "" if not force else " --force"
    for pref in ["H", "V"]:
        targets = os.path.join(export_dir, pref, pref + "_*_*_meters.tif")
        cmd = '''deramp_ts.py "{}" --ramp=linear --ramp-opts="{}" --date-offset=1'''.format(
                targets,
                ramp_opts
            ) + add
        logger.info(cmd)
        sh(cmd)

def ensure_directories(export_dir, force):
    def check_dir(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    dirs = [
        export_dir,
        os.path.join(export_dir, 'H'),
        os.path.join(export_dir, 'V'),
        os.path.join(export_dir, 'NCC')
    ]

    for d in dirs:
        check_dir(d)



if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    if arguments["--verbose"]:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s: %(asctime)s | %(message)s"
        )

    stereo_dir = arguments["<stereo>"]
    export_dir = arguments["<export>"]
    pairs_file = arguments["--pairs"]
    ramp = arguments["--ramp"]
    res = arguments["--tr"]
    force = arguments["--force"]

    pairs = Pair.read_from_file(pairs_file)
    missing_f = []
    missing_ncc = []
    for p in pairs:
        # find files
        p.search_stereo(stereo_dir, missing_f, missing_ncc)

    np.savetxt(os.path.join(export_dir, "missing_f.txt"), missing_f, "%s")
    np.savetxt(os.path.join(export_dir, "missing_ncc.txt"), missing_ncc, "%s")

    ensure_directories(export_dir, force)

    # stereo samplings
    h_res, v_res = np.loadtxt('sampling.txt', unpack=True, skiprows=1)
    
    for p in pairs:
        # Fetch
        p.link_or_resample(export_dir, res=res)
        # Apply ndv and convert to meters 
        p.apply_corrections(export_dir, [h_res, v_res], force)
    
    # Deramp using time series ramps
    apply_ramps(export_dir, ramp, force)
    # TODO skip all before if ramp output already exists or force it

    for p in pairs:
        # TODO skip if r4 already exists or force it
        p.remove_median(export_dir)
        p.switch_to_envi(export_dir)
        #p.clean_files(export_dir)
