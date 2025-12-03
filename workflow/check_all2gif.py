#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_all2gif.py
-----------
Check the ALL2GIF.sh results before processing with ASP

Usage: check_all2gif.py <all2gif>
check_all2gif.py -h | --help

Options:
-h --help       Show this screen
--check         Path to ALL2GIF.sh results


"""

import os
import docopt


def check_for_empty_files(all2gif_dir, do_print=False):
    out = []
    refcol = None
    refrow = None
    is_empty = True
    for d in os.listdir(all2gif_dir):
        if(os.path.isdir(os.path.join(all2gif_dir, d)) and d[0] == '2'):
            is_empty = False
            insar_param_file = os.path.join(all2gif_dir, d, 'i12', 'TextFiles', 'InSARParameters.txt')
            with open(insar_param_file, 'r') as f:
                lines = [''.join(l.strip().split('\t\t')[0]) for l in f.readlines()]
                jump_index = lines.index('/* -5- Interferometric products computation */')
                img_dim = lines[jump_index + 2: jump_index + 4]

                ncol, nrow = int(img_dim[0].strip()), int(img_dim[1].strip())
                if ncol == 0 or nrow == 0:
                    if do_print:
                        print(">> EMPTY:", d)
                    out += [d]
                elif refcol is not None and refrow is not None:
                    if ncol != refcol or nrow != refrow:
                        if do_print:
                            print('>> INCONSISTENT SIZE: ({}, {}):'.format(ncol, nrow), d)
                        out += [d]
                    else:
                        print("ok: {}".format(d))
                else:
                    refcol = ncol
                    refrow = nrow
                    if do_print:
                        print("REF SIZE: ({}, {})".format(ncol, nrow))
                    print("ok: {}".format(d))
    if is_empty:
        print("There is no target in this folder. Take a deep breath and try again in another folder")
    return out


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    all2gif_dir = arguments["<all2gif>"]

    empty_files = check_for_empty_files(all2gif_dir, do_print=True)

    if(not empty_files):
        print('Everything should be fine, can continue with process_stereo')
    else:
        print('Check following directories; adjust LLRGCO & LLAZCO and process ALL2GIF again')
        for d in empty_files:
            print(os.path.join(all2gif_dir, d))
