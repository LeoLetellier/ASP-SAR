#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_star.py

Usage: merge_star.py <expr> --output=<output>
merge_star.py -h | --help

Options:
-h | --help             Show this screen
"""

import docopt
import os, sys, subprocess
import glob

def sh(cmd: str, shell=True):
    subprocess.run(cmd, shell=shell, stdout=sys.stdout, stderr=subprocess.STDOUT, env=os.environ)

arguments = docopt.docopt(__doc__)
expr = arguments["<expr>"]
output = arguments["--output"]

files = glob.glob(expr)
files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))
print(files)
sh("gdal_merge.py -o {} -separate {}".format(output, " ".join(files)))
