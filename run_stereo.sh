#!/usr/bin/bash

#module load asp

# correl_dir is path to current run dir (data_dir/CORREL)
DATA_DIR=$1
CORREL_DIR=$2
DATE1=$3
DATE2=$4
# name of directory for pair processing results f.e. 20220717_20221104
PAIR=$DATE1"_"$DATE2

# load asp_parameters.txt in DATA_DIR (DATA_DIR = WORK_DIR)
. $DATA_DIR"/asp_parameters.txt"

IMG_PRE=$DATA_DIR"/GEOTIFF/"$3".VV.mod_log.tif"
IMG_POST=$DATA_DIR"/GEOTIFF/"$4".VV.mod_log.tif"

cd $CORREL_DIR

mkdir $PAIR

cd $PAIR

session="-t $SESSION_TYPE --alignment-method $A_M --threads-multiprocess $THREADS"
stereo="--corr-kernel $CORR_KERNEL --cost-mode $COST_MODE --stereo-algorithm $ST_ALG --subpixel-mode $SUBP_MODE --subpixel-kernel $SUBP_KERNEL --sgm-collar-size $SGM_C_SIZE"
denoising="--corr-blob-filter 3  --prefilter-mode 1 --prefilter-kernel-width 35"

parallel_stereo $session $IMG_PRE $IMG_POST $BLACK_LEFT $BLACK_RIGHT $OUTPUT_DIR $stereo $denoising $filtering --stop-point 5

exit

