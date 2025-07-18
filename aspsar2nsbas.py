#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster2aspsar.py
--------------
Prepare the directory structure for further processing. Link all ALL2GIF results in the given destination dir.

Usage: amster2aspsar.py --aspsar=<path> [--s1]
amster2aspsar.py -h | --help

Options:
-h | --help         Show this screen
--aspsar              Path to aspsar processing directory
--s1

"""

import docopt

from workflow.prepare_result_export import *
from workflow.prepare_nsbas_process import *


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    pass

    #prepare_result_export
    #prepare_nsbas_process
    #mask ?
