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
    files.sort()
    names = [os.path.basename(f) for f in files]
    dates_str = [n[:8] for n in names]
    dates_dt = [datetime.strptime(dstr, "%Y%m%d") for dstr in dates_str]

    if not os.path.isdir(folder + "/STACKTIF"):
        os.mkdir(folder + "/STACKTIF")

    central_stacks = []
    central_date = []

    for i, d in enumerate(tqdm(dates_dt)):
        stack_name = []
        stack_date_str = []
        outfile = folder + "/STACKTIF/" + str(names[i][:8]) + "_stack" + str(int(minimal_bt)) + ".tif"
        if not os.path.isfile(outfile):
            forwards = []
            backwards = []
            for i2, d2 in enumerate(dates_dt):
                # print(d, d2, abs((d2 - d).days), abs((d2 - d).days) < minimal_bt / 2)
                # if abs((d2 - d).days) < minimal_bt / 2:
                #     stack_name.append(names[i2])
                #     stack_date_str.append(dates_str[i2])
                bt = (d2 - d).days
                if 0 < bt < minimal_bt / 2:
                    # print("check", names[i2])
                    forwards.append([names[i2], abs(bt)])
                    stack_date_str.append(dates_str[i2])
                elif 0 > bt > - minimal_bt / 2:
                    backwards.append([names[i2], abs(bt)])
                    stack_date_str.append(dates_str[i2])
            forwards.sort(key=lambda p: p[1])
            backwards.sort(key=lambda p: p[1])
            forwards = forwards[:2]
            forwards = [f[0] for f in forwards]
            backwards = backwards[:2]
            backwards = [f[0] for f in backwards]
            stack_name = forwards + backwards + [names[i]]
            # retained = [r.split(".")[0] for r in retained]
            stack_name.sort()
            if len(stack_name) > 0:
                central_date.append(dates_str[i])
                central_stacks.append([r.split(".")[0] for r in stack_name])
            # if len(stack_date_str) > 0:
            #     central_date.append(dates_str[i])
            # #     central_stacks.append(stack_date_str)
            #     print(i)
            #     print(central_date[-1])
            #     print(central_stacks[-1])
            #     print(len(central_stacks[-1]))
            #     print()
            cmd = "stack_median.py {} --outfile={}".format(
                " ".join([folder + "/GEOTIFF/" + s for s in stack_name]),
                outfile
            )
            # print(cmd)
            sh(cmd)

    # print("stacks", central_stacks[:3])
    # print("stack_date", stack_date_str[:3])
    
    if len(central_date) != len(central_stacks):
        raise ValueError("Central date and central stacks should have same size but does not: {}!={}".format(
            len(central_date),
            len(central_stacks)
        ))

    with open(os.path.join(folder, "STACKTIF/stack_images.txt"), 'w') as f:
        content = [central_date[i] + "\t" + str(len(central_stacks[i])) + "\t" + " ".join(s) for i, s in enumerate(central_stacks)]
        f.write("\n".join(content))
