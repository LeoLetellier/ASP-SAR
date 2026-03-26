#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################
#
# PyGdalSAR: An InSAR post-processing package
# written in Python-Gdal
#
############################################
# Author        : Simon DAOUT (Oxford)
############################################
# Adapted       : Leo LETELLIER (CRPG)
############################################


"""\
invert_pair2date.py
-------------
Invert per dates a property value defined at pairs

Usage: invert_pair2date.py --date1=<date1> --date2=<date2> --values=<values> --outfile=<outfile> [--noise]
invert_pair2date.py  -h | --help

Options:
-h --help           Show this screen.

Adapted from [PyGdalSAR](https://github.com/simondaout/PyGdalSAR/blob/master/NSBAS-playground/sandbox/invert_phi.py)
"""

import docopt
import scipy.linalg as lst
import scipy.optimize as opt
import numpy as np


def consInvert(A, b, sigmad=1, ineq=[None, None], cond=1.0e-10, iter=250, acc=1e-06):
    """Solves the constrained inversion problem.

    Minimize:

    ||Ax-b||^2

    Subject to:
    Ex >= f
    """

    Ain = A
    bin = b

    if Ain.shape[0] != len(bin):
        raise ValueError("Incompatible dimensions for A and b")

    Ein = ineq[0]
    fin = ineq[1]

    if Ein is not None:
        if Ein.shape[0] != len(fin):
            raise ValueError("Incompatible shape for E and f")
        if Ein.shape[1] != Ain.shape[1]:
            raise ValueError("Incompatible shape for A and E")

    ####Objective function and derivative
    _func = lambda x: np.sum(((np.dot(Ain, x) - bin) / sigmad) ** 2)
    _fprime = lambda x: 2 * np.dot(Ain.T / sigmad, (np.dot(Ain, x) - bin) / sigmad)

    ######Inequality constraints and derivative
    if Ein is not None:
        _f_ieqcons = lambda x: np.dot(Ein, x) - fin
        _fprime_ieqcons = lambda x: Ein

    ######Actual solution of the problem
    temp = lst.lstsq(Ain, bin, cond=cond)  ####Initial guess.
    x0 = temp[0]

    if Ein is None:
        res = temp
    else:
        res = opt.fmin_slsqp(
            _func,
            x0,
            f_ieqcons=_f_ieqcons,
            fprime=_fprime,
            fprime_ieqcons=_fprime_ieqcons,
            iter=iter,
            full_output=True,
            acc=acc,
        )
        if res[3] != 0:
            print("Exit mode %d: %s \n" % (res[3], res[4]))

    fsoln = res[0]
    return fsoln


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    date1_file, date1_col = arguments["--date1"].split(",")
    date2_file, date2_col = arguments["--date2"].split(",")
    values_wanted = arguments["--values"].split(",")
    value_file, value_col = values_wanted[0], values_wanted[1:]
    outfile = arguments["--outfile"]

    date1 = np.loadtxt(date1_file, usecols=int(date1_col), dtype=str, unpack=True)
    date2 = np.loadtxt(date2_file, usecols=int(date2_col), dtype=str, unpack=True)
    values = np.loadtxt(value_file, usecols=[int(v) for v in value_col], dtype=float, unpack=True)
    ncoeffs = len(values)

    dates = list(set(date1.tolist() + date2.tolist()))
    dates.sort()

    ndates = len(dates)
    npairs = len(date1)

    if len(date2) != npairs or len(values[0]) != npairs:
        raise ValueError("size mismatch")

    G = np.zeros((npairs + 1, ndates))
    if arguments["--noise"]:
        for k in range((npairs)):
            for n in range((ndates)):
                if date1[k] == dates[n]:
                    G[k, n] = 1
                elif date2[k] == dates[n]:
                    G[k, n] = 1
    else:
        for k in range((npairs)):
            for n in range((ndates)):
                if date1[k] == dates[n]:
                    G[k, n] = -1
                elif date2[k] == dates[n]:
                    G[k, n] = 1
    # init first to 0
    G[-1, 0] = 1

    # build d
    d = np.zeros((npairs + 1, ncoeffs))
    d[:len(date1), :] = np.column_stack(values)

    print("Inversion....")
    sp = consInvert(G, d)

    print(np.array(dates)[:, np.newaxis].shape, sp.shape)
    print(dates, sp)

    print("Saving in the output file", outfile)
    np.savetxt(outfile, np.concatenate([np.array(dates)[:, np.newaxis], sp], axis=1), fmt="%s")
