#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_pairs.py
------------

Usage: figure_pairs.py <aspsar> [--ref=<ref> --table=<table>]

Options:
-h | --help         Show this screen
"""

import docopt
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from cmcrameri import cm
import matplotlib as mpl

gdal.UseExceptions()


class Pair:
    def __init__(self, date1, date2):
        if date1 > date2:
            self.date1 = datetime.strptime(date2, "%Y%m%d")
            self.date2 = datetime.strptime(date1, "%Y%m%d")
        else:
            self.date1 = datetime.strptime(date1, "%Y%m%d")
            self.date2 = datetime.strptime(date2, "%Y%m%d")
        self.bt = self._compute_bt()
        wet_period = ["06", "07", "08", "09", "10"]
        self.is_wet = date1[4:6] in wet_period or date2[4:6] in wet_period
        self.mean_h = None
        self.sigma_h = None
        self.mean_v = None
        self.sigma_v = None
        self.mean_ncc = None
        self.sigma_ncc = None
        self.corr_px = None

    def _compute_bt(self):
        self.bt = (self.date2 - self.date1).days
        return self.bt
    
    def __eq__(self, value):
        return self.date1 == value.date1 and self.date2 == value.date2
    
    def __lt__(self, other):
        if self.date1 == other.date1:
            return self.date2 <= other.date2
        return self.date1 <= other.date1

    def to_line(self):
        return [self.date1.strftime("%Y%m%d"), self.date2.strftime("%Y%m%d"), str(self.bt), str(self.mean_h), str(self.sigma_h), str(self.mean_v), str(self.sigma_v), str(self.mean_ncc), str(self.sigma_ncc), str(self.corr_px)]
    
    @staticmethod
    def from_line(line):
        p = Pair(line[0], line[1])
        p.mean_h = float(line[3])
        p.sigma_h = float(line[4])
        p.mean_v = float(line[5])
        p.sigma_v = float(line[6])
        p.mean_ncc = float(line[7])
        p.sigma_ncc = float(line[8])
        p.corr_px = float(line[9])
        return p


def raster_mean_sigma(data, do_cpx=False):
    data = data.flatten()
    mean = np.nanmean(data)
    sigma = np.nanstd(data)
    if do_cpx:
        corr_px = np.count_nonzero(~np.isnan(data)) / len(data)
        return mean, sigma, corr_px
    return mean, sigma


def save_pair_data(file, pairs):
    content = "d1;d2;bt;mh;sh;mv;sv;mncc;sncc;cpx\n"
    for p in pairs:
        content += ';'.join(p.to_line())
        content += '\n'
    with open(file, 'w') as outfile:
        outfile.write(content)
    print("Saved:", file)


def load_pair_data(file):
    data = np.loadtxt(file, delimiter=';', skiprows=1, dtype=str)
    pairs = []
    for l in data:
        p = Pair.from_line(l)
        pairs.append(p)
    return pairs


def read_pairs_file(file):
    pairs = np.loadtxt(file, skiprows=2, usecols=(0, 1), delimiter='\t', dtype=str)
    pairs = [Pair(p[0], p[1]) for p in pairs]
    return pairs


def open_gdal(file, band=1, supp_ndv=None, crop=None):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    if crop is None:
        data = band.ReadAsArray()
    else:
        data = band.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])
    if ndv is not None and ndv != np.nan:
        data[data==ndv] = np.nan
    if supp_ndv is not None and supp_ndv != np.nan:
        data[data==supp_ndv] = np.nan
    return data


def open_r4(file, crop=None):
    rsc = file + '.rsc'
    lines = open(rsc).read().strip().split('\n')
    x_dim, y_dim = None, None
    for l in lines:
        if 'WIDTH' in l:
            x_dim = int(''.join(filter(str.isdigit, l)))
        elif 'FILE_LENGTH' in l:
            y_dim = int(''.join(filter(str.isdigit, l)))
        if x_dim is not None and y_dim is not None:
            break
    data = np.fromfile(file, dtype=np.float32)[:y_dim * x_dim].reshape((y_dim, x_dim))
    if crop is not None:
        data = data[crop[2]:crop[3], crop[0]:crop[1]]
    return data


def plot_raster(data, title, ax):
    vmin, vmax = np.nanpercentile(data, (2, 98))
    ims = ax.imshow(data, "Greys_r", interpolation="nearest", vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ims, cax=c)
    ax.set_title(title)


def compute_mean_da(folder, subset):
    geotiff = os.path.join(folder, "GEOTIFF")
    rasters = [f for f in os.listdir(geotiff) if f[-len('.mod_log.tif'):] == ".mod_log.tif" and f[:8] in subset]
    shape = open_gdal(os.path.join(folder, 'GEOTIFF', rasters[0])).shape
    
    mean = np.zeros(shape=shape)
    sigma = np.zeros(shape=shape)
    
    for r in tqdm(rasters):
        data = open_gdal(os.path.join(folder, 'GEOTIFF', r))
        mean += data
        sigma += np.square(data)
    
    mean /= len(rasters)
    sigma = np.sqrt(sigma - np.square(mean))
    da = sigma / mean

    return mean, da


def savefig(path):
    plt.tight_layout()
    print("Saving:", path)
    plt.savefig(path, dpi=300)


def compute_std_area(rasters, area):
    stds = []
    for r in rasters:
        data = open_gdal(r, crop=area)
        stds.append(np.nanstd(data))
    return stds


def consInvert(A,b,sigmad=1,ineq=[None,None], cond=1.0e-10, iter=250,acc=1e-06):
    '''Solves the constrained inversion problem.

    Minimize:
    
    ||Ax-b||^2

    Subject to:
    Ex >= f
    '''

    import scipy.linalg as lst
    import scipy.optimize as opt
    
    Ain = A
    bin = b

    if Ain.shape[0] != len(bin):
        raise ValueError('Incompatible dimensions for A and b')

    Ein = ineq[0]
    fin = ineq[1]

    if Ein is not None:
        if Ein.shape[0] != len(fin):
            raise ValueError('Incompatible shape for E and f')
        if Ein.shape[1] != Ain.shape[1]:
            raise ValueError('Incompatible shape for A and E')

    ####Objective function and derivative
    _func = lambda x: np.sum(((np.dot(Ain,x)-bin)/sigmad)**2)
    _fprime = lambda x: 2*np.dot(Ain.T/sigmad, (np.dot(Ain,x)-bin)/sigmad)

    ######Inequality constraints and derivative
    if Ein is not None:
        _f_ieqcons = lambda x: np.dot(Ein,x)-fin
        _fprime_ieqcons = lambda x: Ein

    ######Actual solution of the problem
    temp = lst.lstsq(Ain,bin,cond=cond)   ####Initial guess.
    x0 = temp[0]

    if Ein is None:
        res = temp
    else:
        res = opt.fmin_slsqp(_func,x0,f_ieqcons=_f_ieqcons,fprime=_fprime, fprime_ieqcons=_fprime_ieqcons,iter=iter,full_output=True,acc=acc)
        if res[3] != 0:
            print('Exit mode %d: %s \n'%(res[3],res[4]))

    fsoln = res[0]
    return fsoln


def invert(date1, date2, value, noise=False):
    G=np.zeros((kmax+1,nmax))
    if noise=='yes':
        for k in range((kmax)):
            for n in range((nmax)):
                if (date1[k]==im[n]): 
                    G[k,n]=1
                elif (date2[k]==im[n]):
                    G[k,n]=1
    else:
        for k in range((kmax)):
            for n in range((nmax)):
                if (date1[k]==im[n]): 
                    G[k,n]=-1
                elif (date2[k]==im[n]):
                    G[k,n]=1
    # ini phi first image to 0 
    G[-1,0]=1

    #build d
    d=np.zeros((kmax+1))
    d[:kmax]=spint


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    folder = arguments["<aspsar>"]
    ref = arguments["--ref"]
    if ref is not None:
        ref = [int(k) for k in ref.split(',', 4)]
    table = arguments["--table"]
    if table is None:
        table = os.path.join(folder, 'PAIRS', 'table_pairs.txt')

    pairs = read_pairs_file(table)

    # First Graphic (/Pair):
    # need mean/sigma per raster for H/V/NCC and % correlated px per raster
    # Scatter plot - x: Bt, y: mean, size: sigma, color: period
    print("> Pair Graphic")
    print("Retrieving data")
    DATA1 = os.path.join(folder, 'FIGURES', "data_pairs_image.txt")
    if not os.path.isfile(DATA1):
        for p in tqdm(pairs):
            f = 'H'
            file = os.path.join(folder, 'EXPORT', f, f + '_' + p.date1.strftime("%Y%m%d") + '_' + p.date2.strftime("%Y%m%d") + '.r4')
            data = open_r4(file)
            m, s = raster_mean_sigma(data)
            p.mean_h = m
            p.sigma_h = s
            f = 'V'
            file = os.path.join(folder, 'EXPORT', f, f + '_' + p.date1.strftime("%Y%m%d") + '_' + p.date2.strftime("%Y%m%d") + '.r4')
            data = open_r4(file)
            m, s = raster_mean_sigma(data)
            p.mean_v = m
            p.sigma_v = s
            f = 'NCC'
            file = os.path.join(folder, 'EXPORT', f, f + '_' + p.date1.strftime("%Y%m%d") + '_' + p.date2.strftime("%Y%m%d") + '.r4')
            data = open_r4(file)
            m, s, corr_px = raster_mean_sigma(data, do_cpx=True)
            p.mean_ncc = m
            p.sigma_ncc = s
            p.corr_px = corr_px
        save_pair_data(DATA1, pairs)
    else:
        pairs = load_pair_data(DATA1)
    pairs.sort()

    if ref is not None:
        pairs_ref = deepcopy(pairs)
        DATA1_CROP = os.path.join(folder, 'FIGURES', "data_pairs_image" + '_'.join([str(k) for k in ref]) + ".txt")
        if not os.path.isfile(DATA1_CROP):
            for p in tqdm(pairs_ref):
                f = 'H'
                file = os.path.join(folder, 'EXPORT', f, f + '_' + p.date1.strftime("%Y%m%d") + '_' + p.date2.strftime("%Y%m%d") + '.r4')
                data = open_r4(file, crop=ref)
                m, s = raster_mean_sigma(data)
                p.mean_h = m
                p.sigma_h = s
                f = 'V'
                file = os.path.join(folder, 'EXPORT', f, f + '_' + p.date1.strftime("%Y%m%d") + '_' + p.date2.strftime("%Y%m%d") + '.r4')
                data = open_r4(file, crop=ref)
                m, s = raster_mean_sigma(data)
                p.mean_v = m
                p.sigma_v = s
                f = 'NCC'
                file = os.path.join(folder, 'EXPORT', f, f + '_' + p.date1.strftime("%Y%m%d") + '_' + p.date2.strftime("%Y%m%d") + '.r4')
                data = open_r4(file, crop=ref)
                m, s, corr_px = raster_mean_sigma(data, do_cpx=True)
                p.mean_ncc = m
                p.sigma_ncc = s
                p.corr_px = corr_px
            save_pair_data(DATA1_CROP, pairs_ref)
        else:
            pairs_ref = load_pair_data(DATA1_CROP)
        pairs_ref.sort()

    print("Plotting")
    pairs = pairs_ref
    
    d1 = np.zeros(len(pairs), dtype=datetime)
    d2 = np.zeros(len(pairs), dtype=datetime)
    bt = np.zeros(len(pairs))
    mean_v = np.zeros(len(pairs))
    sigma_v = np.zeros(len(pairs))
    mean_h = np.zeros(len(pairs))
    sigma_h = np.zeros(len(pairs))
    wet = np.zeros(len(pairs))
    cpx = np.zeros(len(pairs))
    ncc_mean = np.zeros(len(pairs))
    ncc_sigma = np.zeros(len(pairs))

    for p in range(len(pairs)):
        d1[p] = pairs[p].date1
        d2[p] = pairs[p].date2
        bt[p] = pairs[p].bt
        mean_v[p] = pairs[p].mean_v
        sigma_v[p] = pairs[p].sigma_v
        mean_h[p] = pairs[p].mean_h
        sigma_h[p] = pairs[p].sigma_h
        wet[p] = pairs[p].is_wet
        cpx[p] = pairs[p].corr_px
        ncc_mean[p] = pairs[p].mean_ncc
        ncc_sigma[p] = pairs[p].sigma_ncc
    
    common_year = 2000
    yearly_d1 = [datetime(common_year, date.month, date.day) for date in d1]
    yearly_d2 = [datetime(common_year, date.month, date.day) for date in d2]

    plt.scatter(bt, sigma_v, label='std')
    plt.scatter(bt, cpx, label='%corr')
    plt.scatter(bt, ncc_mean, label='NCC')
    plt.legend()
    plt.figure()

    from matplotlib.colors import LogNorm
    from matplotlib.colors import BoundaryNorm, ListedColormap
    bounds = [0, 60, 150, 330, 390, 3653]
    cmap = ListedColormap([cm.navia(i) for i in np.linspace(0, 1, len(bounds)-1)])
    norm = BoundaryNorm(bounds, cmap.N)
    plt.scatter(d1, sigma_h, s=bt/3, c=bt, norm=norm, cmap=cmap, edgecolors='black', alpha=0.9, linewidths=0.7)
    plt.scatter(d2, sigma_h, s=bt/3, c=bt, norm=norm, cmap=cmap, edgecolors='black', alpha=0.9, linewidths=0.7)
    plt.xlabel("Date")
    plt.ylabel("STD (m)")
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(["0 month", "3 months", "5 months", "11 months", "13 months", "10 years"])
    plt.figure()
    # plt.scatter(yearly_d1, sigma_v, s=bt/3, c=bt, norm=LogNorm(), cmap=cm.navia, edgecolors='black', alpha=0.9, linewidths=0.7)
    plt.scatter(yearly_d1, sigma_h, s=bt/3, c=bt, norm=norm, cmap=cmap, edgecolors='black', alpha=0.9, linewidths=0.7)
    plt.scatter(yearly_d2, sigma_h, s=bt/3, c=bt, norm=norm, cmap=cmap, edgecolors='black', alpha=0.9, linewidths=0.7)
    plt.ylabel("STD (m)")
    plt.xlabel("Yearly date")
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(["0 month", "3 months", "5 months", "11 months", "13 months", "10 years"])

    plt.show()
    plt.figure()

    # fig, ax = plt.subplots()
    # cmap = cm.navia
    # cmap = plt.cm.colors.ListedColormap([cmap(0.7), cmap(1)])
    # size = 1 + (mean_h + 1) * 15
    # ax.scatter(bt, sigma_h, s=size, c=wet, linewidths=mean_h, cmap=cmap)
    # norm = mpl.colors.BoundaryNorm([0, 1, 2], cmap.N)
    # divider = make_axes_locatable(ax)
    # c = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #          cax=c, orientation='vertical',
    #          label="Temporal Baselines (month)",
    #          fraction=0.02, pad=0.04)
    # plt.show()


    target_velocities = [0.06, 0.12, 0.25, 0.5, 1, 2]
    # target_velocities = [0.01, 0.5, 2]
    daily_disp = [k / 365.25 for k in target_velocities]

    plt.scatter(bt, sigma_h / np.sqrt(50))
    plt.scatter(bt, sigma_v / np.sqrt(50))
    btmax = np.max(bt)
    for i, k in enumerate(daily_disp):
        # plt.fill_between([0, btmax], [0, btmax * k], [0, btmax * k * 5], alpha=0.2)
        plt.plot([0, btmax], [0, btmax * k], label=str(target_velocities[i]) + "m/yr")

    plt.legend()
    plt.show()

    # plt.figure()
    # color = cm.batlow(bt/np.max(bt))
    # plt.scatter(d1, cpx, color=color)

    # plt.figure()
    # color = cm.batlow(bt/np.max(bt))
    # plt.scatter(d2, cpx, color=color)
    # plt.show()

    # plt.figure()
    # plt.scatter(bt, cpx, label='cpx')
    # # plt.xscale('log')
    # plt.legend()

    # plt.figure()
    # plt.scatter(bt, ncc_mean)
    # ax = plt.gca().twinx()
    # ax.scatter(bt, ncc_sigma, color='red')
    # # plt.xscale('log')
    # plt.legend()

    # plt.figure()
    # plt.scatter(bt[wet==0], mean_h[wet==0], marker='o', label='mean_h')
    # plt.scatter(bt[wet==1], mean_h[wet==1], marker='s', label='mean_h')
    # plt.scatter(bt, mean_v, label='mean_v')
    # # plt.xscale('log')
    # plt.legend()

    # plt.figure()
    # plt.scatter(bt, sigma_h, label='sigma_h')
    # plt.scatter(bt, sigma_v, label='sigma_v')
    # # plt.xscale('log')
    # plt.legend()
    # plt.show()

    # bt = np.zeros(len(pairs_ref))
    # mean_v = np.zeros(len(pairs_ref))
    # sigma_v = np.zeros(len(pairs_ref))
    # mean_h = np.zeros(len(pairs_ref))
    # sigma_h = np.zeros(len(pairs_ref))
    # wet = np.zeros(len(pairs_ref))
    # cpx = np.zeros(len(pairs_ref))
    # ncc_mean = np.zeros(len(pairs_ref))
    # ncc_sigma = np.zeros(len(pairs_ref))

    # for p in range(len(pairs_ref)):
    #     bt[p] = pairs_ref[p].bt
    #     mean_v[p] = pairs_ref[p].mean_v
    #     sigma_v[p] = pairs_ref[p].sigma_v
    #     mean_h[p] = pairs_ref[p].mean_h
    #     sigma_h[p] = pairs_ref[p].sigma_h
    #     wet[p] = pairs_ref[p].is_wet
    #     cpx[p] = pairs_ref[p].corr_px
    #     ncc_mean[p] = pairs_ref[p].mean_ncc
    #     ncc_sigma[p] = pairs_ref[p].sigma_ncc
    

    # plt.figure()
    # plt.scatter(bt, cpx, label='cpx')
    # # plt.xscale('log')
    # plt.legend()

    # plt.figure()
    # plt.scatter(bt, ncc_mean, label='cpx')
    # plt.scatter(bt, ncc_sigma, label='cpx')
    # plt.legend()

    # plt.figure()
    # plt.scatter(bt, mean_h, marker='o', label='mean_h')
    # # plt.scatter(bt[wet==1], mean_h[wet==1], marker='s', label='mean_h')
    # plt.scatter(bt, mean_v, label='mean_v')
    # plt.legend()

    # plt.figure()
    # plt.scatter(bt, sigma_h, label='sigma_h')
    # plt.scatter(bt, sigma_v, label='sigma_v')
    # plt.legend()
    # plt.show()

    # fig, ax = plt.subplots()
    # cmap = cm.navia
    # cmap = plt.cm.colors.ListedColormap([cmap(0.7), cmap(1)])
    # ax.scatter(bt, sigma_h, s=mean_h * 5 + 0.8, c=wet, linewidths=mean_h, cmap=cmap)
    # norm = mpl.colors.BoundaryNorm([0, 1, 2], cmap.N)
    # divider = make_axes_locatable(ax)
    # c = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #          cax=c, orientation='vertical',
    #          label="Temporal Baselines (month)",
    #          fraction=0.02, pad=0.04)
    # plt.show()


    # Second Graphic (/Pixel):
    # need mean/sigma per raster for H/V/NCC and % correlated px per pixel
    # 7 raster plots for each
    
    
