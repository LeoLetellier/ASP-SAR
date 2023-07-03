# help script to create default asp_parameters.txt in dst_dir

# SYNTAX init_asp_parameters.py [DESTINATION PATH]

import os, sys


def init_asp_parameters(correl_path):

    black_img_dir = os.path.join(os.getcwd(), 'contrib', 'data')
    asp_parameter_content = """
####################
## PARAMETER FILE ##
####################

# INFO: 

#####################
# GENERAL VARIABLES #
#####################

THREADS="10"

####################
# SET INPUT IMAGES #
####################

# in new version, path is changed 
BLACK_LEFT="{}/black_left.tsai"
BLACK_RIGHT="{}/black_right.tsai"

###################
# STEREO SETTINGS #
###################

SESSION_TYPE="pinhole" # -t
A_M="none" # --alignment-method
DATUM="wgs84" # --datum
OUTPUT_DIR="asp/correl" #creates dir within working dir - every created dataset starts with correl-
NO_DATA_S="0" # --nodata_value stereo
CORR_KERNEL="7 7" # --corr_kernel
COST_MODE="3" # --cost-mode
ST_ALG="asp_mgm" # --stereo_algorithm
CORR_T_S="1024" # --corr-tile-size
SUBP_MODE="9" # --subpixel-mode
SUBP_KERNEL="5 5" # --subpixel-kernel
CORR_S_MODE="1" # --corr_seed_mode

XCORR_TH="2.0" # --xcorr-threshold
MIN_XCORR_LVL="0" # --min-xcorr-level
SGM_C_SIZE="512" # --sgm-collar-size

# 31.03 added
PREF_MODE="2" # --prefilter-mode
PREF_KER_M="1.5" # --prefilter-kernel-width

# Filtering #

FILTERING=false

RM_QUANT_PC="0.8" # --rm-quantile-percentile
RM_QUANT_MULT="1" # --rm-quantile-multiple
RM_CLEAN_PASS="2" # --rm-cleanup-passes
FILTER_MODE="1" # --filter-mode
RM_HALF_KERN="5 5" # --rm-half-kernel
# 29.03 added
RM_MIN_MAT="50" # --rm-min-matches
RM_TH="3" # --rm-threshold


""".format(black_img_dir, black_img_dir)

    with open(os.path.join(correl_path, 'asp_parameters.txt'), 'w') as f:
        f.write(asp_parameter_content)

