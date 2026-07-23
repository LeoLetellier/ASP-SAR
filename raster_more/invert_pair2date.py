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
-------------------
Invert per dates a property value defined at pairs.

If the property is an evolution (differential) between two dates (Pb - Pa), use it as is. Otherwise\
 if it represents an additive value between the two dates, use --noise.

Usage: invert_pair2date.py --date1=<date1> --date2=<date2> --values=<values> --outfile=<outfile> [--noise] [--prop=<prop>] [--cst-w=<w>] [--delimiter=<d>]
invert_pair2date.py  -h | --help

Options:
  -h --help             Show this screen
  --date1=<d>           Path to file and column number to first date of the pair, comma separated (file,nb)
  --date2=<d>           Path to file and column number to second date of the pair, comma separated (file,nb)
  --values=<v>          Path to file and column numbers of properties, comma separated (file,nb1,nb2)
  --outfile=<o>         Text output file to write the results
  --noise               Use the property as additive instead of differential

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


def create_padded_diagonal_array(N, pad):
    size = N - 1
    # Create an M x total_columns array filled with zeros
    arr = np.zeros((N-1, N + pad), dtype=int)

    # Fill the main diagonal with -1
    for i in range(min(N-1, size)):
        arr[i, i] = -1

    # Fill the upper diagonal with 1
    for i in range(min(N-1, size )):
        arr[i, i + 1] = 1

    return arr


def date_inversion(values, date1, date2, dates, additive=False, prop=None, constant_weight= None):
    ndates, npairs, ncoeffs = len(dates), len(date1), len(values)

    if len(date2) != npairs or len(values[0]) != npairs:
        raise ValueError("size mismatch")

    G = np.zeros((npairs + 1, ndates))
    ncols = ndates
    if prop is not None:
        ncols += 1
        G = np.column_stack([G, np.concatenate([prop, [0]])])
    if constant_weight is not None:
        reg = constant_weight * create_padded_diagonal_array(ndates, ndates - ncols)
        G = np.vstack([G, reg])
    if additive:
        # Inversion by additive value
        for k in range((npairs)):
            for n in range((ndates)):
                if date1[k] == dates[n]:
                    G[k, n] = 1
                elif date2[k] == dates[n]:
                    G[k, n] = 1
    else:
        # Inversion by differential value
        for k in range((npairs)):
            for n in range((ndates)):
                if date1[k] == dates[n]:
                    G[k, n] = -1
                elif date2[k] == dates[n]:
                    G[k, n] = 1
    # Init first date to 0
    if not additive:
        G[-1, 0] = 1

    # Build values matrix
    d = np.zeros((npairs + 1, ncoeffs))
    d[:len(date1), :] = np.column_stack(values)
    if constant_weight is not None:
        d = np.vstack([d, np.zeros((ndates - 1, ncoeffs))])

    # Invert
    sp = consInvert(G, d)
    if prop is not None:
        return sp[:-1], sp[-1]
    return sp


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    date1_file, date1_col = arguments["--date1"].split(",")
    date2_file, date2_col = arguments["--date2"].split(",")
    values_wanted = arguments["--values"].split(",")
    value_file, value_col = values_wanted[0], values_wanted[1:]
    prop_file, prop_col = None, None
    if arguments["--prop"] is not None:
        # add a term propotionnal to a pair value
        prop_file, prop_col = arguments["--prop"].split(",")
    outfile = arguments["--outfile"]
    constant_weight = float(arguments["--cst-w"]) if arguments["--cst-w"] is not None else None
    delimiter = arguments["--delimiter"]

    date1 = np.loadtxt(date1_file, usecols=int(date1_col), dtype=str, unpack=True, delimiter=delimiter)
    date2 = np.loadtxt(date2_file, usecols=int(date2_col), dtype=str, unpack=True, delimiter=delimiter)
    values = np.loadtxt(value_file, usecols=[int(v) for v in value_col], dtype=float, unpack=True, delimiter=delimiter)
    prop = None if prop_file is None else np.loadtxt(prop_file, usecols=int(prop_col), dtype=str, unpack=True, delimiter=delimiter)

    dates_initial = list(set(date1.tolist() + date2.tolist()))
    dates_initial.sort()

    # values[np.abs(values) > 20] = np.nan

    # Discard pairs where value is nan
    if values.ndim == 1:
        mask = ~np.isnan(values)
    else:
        mask = ~np.isnan(values).any(axis=0)
    date1 = date1[mask]
    date2 = date2[mask]
    values = values[mask] if values.ndim == 1 else values[:, mask]
    prop = None if prop is None else prop[mask]

    # If there is only one column for values need to simulate a 2D array
    if values.ndim == 1:
        values = np.array([values])

    dates = list(set(date1.tolist() + date2.tolist()))
    dates.sort()

    values_date = date_inversion(values, date1, date2, dates, additive=arguments["--noise"], prop=prop, constant_weight=constant_weight)
    print(values_date)
    if prop is not None:
        values_date, coeff_prop = values_date
        print(f"Corrected inversion with proportionnality coefficient: {coeff_prop}")

    missing_dates = [d for d in dates_initial if d not in dates]
    if len(missing_dates) > 0:
        # Append at the end the missing dates with a nan value
        dates += missing_dates
        # print(values_date.shape, np.array([[np.nan for _ in range(len(missing_dates))]]).shape)
        # values_date = np.concatenate((values_date, np.array([[np.nan for _ in range(len(missing_dates))]])))
        nan_rows = np.full((len(missing_dates), values_date.shape[1]), np.nan)
        values_date = np.concatenate((values_date, nan_rows), axis=0)

    print("Saving in the output file", outfile)
    np.savetxt(outfile, np.concatenate([np.array(dates)[:, np.newaxis], values_date], axis=1), fmt="%s")
