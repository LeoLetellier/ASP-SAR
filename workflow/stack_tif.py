import os
from math import *
from tqdm import tqdm
import glob
from datetime import datetime
import subprocess, sys


def sh(cmd: str, shell: bool = True):
    return subprocess.run(
        cmd,
        shell=shell,
        env=os.environ,
    )


def stack_tif(folder, minimal_bt):
    files = glob.glob(folder + "/GEOTIFF/*.tif")
    names = [os.path.basename(f) for f in files]
    dates = [datetime.strptime(n[:8], "%Y%m%d") for n in names]

    if not os.path.isdir(folder + "/STACKTIF"):
        os.mkdir(folder + "/STACKTIF")

    for i, d in tqdm(enumerate(dates)):
        stack = []
        for i2, d2 in enumerate(dates):
            if abs((d2 - d).days) < minimal_bt / 2:
                stack.append(names[i2])
        cmd = "stack_median.py {} --outfile={}".format(
            " ".join([folder + "/GEOTIFF/" + s for s in stack]),
            folder + "/STACKTIF/" + str(names[i][:8]) + "_stack" + str(int(minimal_bt)) + ".tif"
        )
        sh(cmd)
