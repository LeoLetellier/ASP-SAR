#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################
# Derived from plot_raster.py
#
# PyGdalSAR: An InSAR post-processing package 
# written in Python-Gdal
#
############################################
# Author        : Mathieu Volat 
#                 Simon Daout (CRPG-ENSG)
############################################

"""
rplot.py
-------------
Display and Cut image file (.unw/.int/.r4/.tiff)

Usage: rplot.py <infile> [--cpt=<values>] [--crop=<values>] \
[--lectfile=<lectfile> | --lectcube=<lectcube> | --parfile=<parfile> | --amfile=<amfile>] [--rad2mm=<rad2mm>] [--title=<title>] [--wrap=<wrap>] \
[--vmin=<vmin>] [--vmax=<vmax>] [--band=<band>] [--cols=<cols>] [--lines=<lines>] [--zoom=<zoom>] [--histo] [--save | -s] [--ndv=<ndv>] [--stats]


Options:
-h --help               Show this screen.
<infile>                Raster to be displayed 
--crop=<crop>           Crop option ("xmin,xmax,ymin,ymax")
--cpt=<cpt>             Indicate colorscale for phase
--wrap=<wrap>           Wrapped phase between value for unwrapped files 
--lectfile=<lectfile>   Path of the lect.in file for r4 format
--lectcube=<lectcube>   Path to lect.in file containing band metadata
--parfile=<parfile>     Path of the .par file of GAMMA
--amfile=<amfile>       Path of the AMSTer InsarParameter file
--rad2mm=<rad2mm>       Convert data [default: 1]
--title=<title>         Title plot 
--band=<band>           Select band number [default: 1] 
--vmax=<vmax>           Max colorscale [default: 98th percentile]
--vmin=<vmin>           Min colorscale [default: 2th percentile]
--cols=<cols> VALUE     Add marker on pixel column numbers (eg. 200,400,450)
--lines=<lines> VALUE   Add marker on pixel lines numbers  (eg. 1200,1200,3000)
--ndv=<ndv>             Use an additionnal no data value
--zoom=<zoom>           Additionnaly display a zoom of the raster ("xmin,xmax,ymin,ymax")
--histo                 Additionnaly display the raster histogram
--stats                 Display the raster and zoom statistics
--save -s               Save the display to pdf
"""

print()
print()
print('Author: Simon Daout')
print()
print('revised version June 2025 (Leo Letellier)')
print()

import os, sys

try:
    from nsbas import docopt
except:
    import docopt

import numpy as np
from numpy.lib.stride_tricks import as_strided

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import setp
from osgeo import gdal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

gdal.UseExceptions()

EXT = {
    'ROIPAC': [
        '.unw',
        '.hgt',
    ],
    'REAL4': [
        '.r4',
    ],
    'GDAL': [
        '.tif',
        '.bil',
    ],
    'AMSTER': [
    ],
    'GAMMA': [
        '.diff',
    ]
}


def resolve_format(infile):
    """Resolve format based on extension and automatic parameter file detection"""
    ext = os.path.splitext(infile)[1]
    infile = os.path.abspath(infile)

    file_format = None
    for key, items in EXT.items():
        if ext in items:
            file_format = key
            break

    maybe_real4_param = os.path.join(os.path.dirname(infile), "lect.in")
    maybe_amster_param = os.path.join(os.path.dirname(os.path.dirname(infile)), "TextFiles", "InSARParameters.txt")
    has_real4_param = os.path.isfile(maybe_real4_param)
    has_amster_param = os.path.isfile(maybe_amster_param)
    
    if file_format == 'REAL4' or (file_format is None and has_real4_param):
        return 'REAL4', maybe_real4_param
    elif file_format == 'AMSTER' or (file_format is None and has_amster_param):
        return 'AMSTER', maybe_amster_param
    elif file_format == 'GAMMA':
        raise ValueError('To use GAMMA file please provide the par header file')
    
    try:
        gdal.Open(infile)
    except:
        if has_amster_param:
            return 'AMSTER', maybe_amster_param
        raise ValueError('Unsupported file')
    return 'GDAL', None


def open_band_gdal(file, band, crop):
    """ Open as GDAL raster
    crop: xmin, xmax, ymin, ymax
    return: data, driver, x, y, b, dtype
    """
    ds = gdal.Open(file)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    if crop is None:
        crop = [0, band.XSize, 0, band.YSize]
    x_dim = crop[1] - crop[0]
    y_dim = crop[3] - crop[2]
    array = band.ReadAsArray(crop[0], crop[2], x_dim, y_dim)
    if ndv is not None and ndv != np.nan:
        array[array == ndv] = np.nan
    return [array], ds.GetDriver().ShortName, x_dim, y_dim, ds.RasterCount, gdal.GetDataTypeName(band.DataType)


def open_band_real4(file, band, params, crop, cube):
    """Open as REAL4 raster"""
    fid = open(file, 'r')
    if crop is not None:
        print("crop will be ignored with .r4")
    if cube:
        x_dim, y_dim, band_nb = list(map(int, open(params).readline().split(None, 3)[0:3]))
        phi = np.fromfile(fid, dtype=np.float32)[:y_dim * x_dim * band_nb].reshape((y_dim, x_dim, band_nb))
        data = [phi[:, :, band - 1]]
    else:
        x_dim, y_dim = list(map(int, open(params).readline().split(None, 2)[0:2]))
        phi = np.fromfile(fid, dtype=np.float32)[:y_dim * x_dim].reshape((y_dim, x_dim))
        data = [phi]
        band_nb = 1
    data_type = np.float32
    driver = 'REAL4'
    return data, driver, x_dim, y_dim, band_nb, data_type


def open_band_roipac(file, crop, supp_ndv):
    """Open as custom ROIPAC raster (amplitude / phase)"""
    ds = gdal.OpenEx(file, allowed_drivers=["ROI_PAC"])
    if crop is None:
        crop = [0, ds.XSize, 0, ds.YSize]
    x_dim = crop[1] - crop[0]
    y_dim = crop[3] - crop[2]
    driver = ds.GetDriver().ShortName
    band_nb = ds.RasterCount

    phase_band = ds.GetRasterBand(2)
    phase_ndv = phase_band.GetNoDataValue()
    phase_data = phase_band.ReadAsArray(crop[0], crop[2], x_dim, y_dim)
    amp_band = ds.GetRasterBand(1)
    amp_ndv = amp_band.GetNoDataValue()
    amp_data = amp_band.ReadAsArray(crop[0], crop[2], x_dim, y_dim)
    
    if phase_ndv is not None and phase_ndv != np.nan:
        phase_data[phase_data == phase_ndv] = np.nan
    if amp_ndv is not None and amp_ndv != np.nan:
        amp_data[amp_data == amp_ndv] = np.nan
    data = [phase_band, amp_band]
    data_type = gdal.GetDataTypeName(phase_band.DataType)

    if supp_ndv is not None:
        data[0][data[0] == supp_ndv] = np.nan
    
    return data, driver, x_dim, y_dim, band_nb, data_type


def open_band_gamma(file, params, crop):
    """Open as GAMMA raster"""
    try:
        from parsers import gamma as gm
    except:
        ModuleNotFoundError("GAMMA parser not found in python installation. Need gamma from module parsers.")

    if params is not None:
        y_dim, x_dim = gm.readpar(par=params)
        phase = gm.readgamma_int(file, par=params)
    else:
        y_dim, x_dim = gm.readpar()
        phase = gm.readgamma_int(file)
    
    data = [phase]
    driver = 'GAMMA'
    band_nb = 1
    data_type = 'unknown'

    return data, driver, x_dim, y_dim, band_nb, data_type


def open_band_amster(file, params, crop):
    """Open as AMSTer raster"""
    with open(params, 'r') as pfile:
        lines = [''.join(l.strip().split('\t\t')[0]) for l in pfile.readlines()]
        jump_index = lines.index('/* -5- Interferometric products computation */')
        img_dim = lines[jump_index + 2: jump_index + 4]
        y_dim, x_dim = (int(img_dim[1].strip()), int(img_dim[0].strip()))
        band_nb = 1
    
    array = np.fromfile(file, dtype=np.float32)
    data = [array[:x_dim * y_dim].reshape((y_dim, x_dim))]
    driver = 'AMSTer'
    data_type = np.float32

    if crop is not None:
        pass

    return data, driver, x_dim, y_dim, band_nb, data_type


def correct_values_phase(phase, ext, rad2mm, wrap, supp_ndv):
    """Apply corrections to phase values"""
    if supp_ndv is not None:
        phase[phase == supp_ndv] = np.nan

    if rad2mm is not None:
        # scale the values
        phase = phase * rad2mm

    if ext in ['.slc']:
        phase = np.absolute(phase)
    if ext in ['.int', '.flat']:
        phase = np.angle(phase)

    if wrap is not None:
        # simulate wrapped values
        phase = np.mod(phase + wrap, 2 * wrap) - wrap
    
    return phase


def correct_values_amp(amp, ext):
    """Cpply corrections to amplitude values"""
    if ext in ['.int', '.flat', '.diff']:
        amp = np.absolute(amp)
    return amp


def resolve_plot(data, arguments, crop):
    """Manage all displays to be plotted"""
    vmin = arguments["--vmin"]
    if vmin is not None:
        vmin = float(vmin)
    vmax = arguments["--vmax"]
    if vmax is not None:
        vmax = float(vmax)
    if (vmax is None) ^ (vmin is None):
        vmin = -vmin if vmin is not None else vmax
        vmax = -vmax if vmax is not None else vmin
    elif (vmax is None and vmin is None):
        vmin = np.nanpercentile(data[0], 2)
        vmax = np.nanpercentile(data[0], 98)
    
    cpt = arguments["--cpt"]
    if cpt is None:
        try:
            from matplotlib.colors import LinearSegmentedColormap
            cm_locs = os.environ["PYGDALSAR"] + '/contrib/python/colormaps/'
            cpt = LinearSegmentedColormap.from_list('roma', np.loadtxt(cm_locs+"roma.txt"))
            cpt = cpt.reversed()
        except:
            cpt=cm.rainbow

    cols = arguments["--cols"]
    lines = arguments["--lines"]
    cross = None
    if cols is not None and lines is not None:
        cross = [[int(k) for k in cols.split(",")], [int(k) for k in lines.split(",")]]
        if crop is not None:
            cross[0] = [k - crop[0] for k in cross[0]]
            cross[1] = [k - crop[2] for k in cross[1]]

    title = arguments["--title"]
    if title is None:
        title = arguments["<infile>"]

    zoom = arguments["--zoom"]
    if zoom is not None:
        zoom = [int(z) for z in zoom.split(',')]

    # Plot the main dislay (phase)
    plot_raster(data[0], cpt, vmin, vmax, cross, title, zoom)
    
    if len(data) > 1:
        # Plot the secondary display (amplitude)
        plot_raster(data[1], title=title + " [Amplitude]")

    if arguments["--histo"]:
        # Plot all histograms
        plot_histo(data, title, zoom)

    if zoom is not None:
        plot_zoom(data, zoom, cpt, title)
    
    if arguments["--stats"]:
        display_stats(data, zoom)


def plot_raster(raster, cpt, vmin, vmax, cross, title, zoom):
    """Construct the raster display"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    hax = ax.imshow(raster, cpt, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(hax, cax=c)

    if cross is not None:
        for i in range(len(cross[0])):
            ax.scatter(cross[0][i], cross[1][i], marker='x', color='black', s=150.0)
    
    if zoom is not None:
        ax.plot([zoom[0], zoom[0], zoom[1], zoom[1], zoom[0]], 
                 [zoom[2], zoom[3], zoom[3], zoom[2], zoom[2]],
                 "-", color='black', linewidth=1)
    
    plt.tight_layout()


def plot_histo(data, title, zoom):
    """Construct the histogram display"""
    fig = plt.figure(figsize=(5, 5))

    histo_data = [data[0]]
    histo_label = ['Main']
    if len(data) > 1:
        histo_data.append(data[1])
        histo_label.append('Secondary')
    if zoom is not None:
        histo_data.append(data[0][zoom[2]:zoom[3], zoom[0]:zoom[1]])
        histo_label.append('Zoom')
    
    for d, l in zip(histo_data, histo_label):
        lower = np.nanpercentile(d, 0.1)
        upper = np.nanpercentile(d, 99.9)
        hist_values, bin_edges = np.histogram(d[~np.isnan(d)].flatten(), bins=50, range=(lower, upper))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, hist_values / np.sum(hist_values), label=l)
    plt.title(title + " [Histogram]")
    plt.legend()

    # TODO add a box plot view ?


def plot_zoom(data, zoom, cpt, title):
    """Construct the zoom display"""
    zdata = data[0][zoom[2]:zoom[3], zoom[0]:zoom[1]]
    vmin = np.nanpercentile(zdata, 2)
    vmax = np.nanpercentile(zdata, 98)
    plot_raster(zdata, cpt, vmin, vmax, cross=None, title=title + " [ZOOM]", zoom=None)


def display_raster_format(infile, driver, x, y, b, dtype):
    """Display information about the raster reading"""
    print(">  File:", infile)
    print(">  Driver:", driver)
    print(">  Size:", x, y, b)
    print(">  DataType:", dtype)


def display_stats(data, zoom):
    from scipy.stats import describe

    print()
    print("> Stats")
    print("\tMain: ", end='')
    print(describe(data[0], axis=None, nan_policy="omit"))

    if len(data) > 1:
        print("\tSecondary: ", end='')
        print(describe(data[1], axis=None, nan_policy="omit"))
    
    if zoom is not None:
        print("\tZoom: ", end='')
        print(describe(data[0][zoom[2]:zoom[3], zoom[0]:zoom[1]], axis=None, nan_policy="omit"))
    

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    infile = arguments["<infile>"]
    ext = os.path.splitext(infile)[1]

    crop = arguments["--crop"]
    if crop is not None:
        crop = [int(k) for k in crop.split(",")]
    
    roicube = False

    if arguments["--lectfile"] is not None:
        file_format = 'REAL4'
        param_file = arguments["--lectfile"]
    elif arguments["--lectcube"] is not None:
        file_format = 'REAL4'
        param_file = arguments["--lectcube"]
        roicube = True
    elif arguments["--parfile"] is not None:
        file_format = 'GAMMA'
        param_file = arguments["--parfile"]
    elif arguments["--amfile"] is not None:
        file_format = 'AMSTER'
        param_file = arguments["--amfile"]
    else:
        file_format = None
        param_file = None

    if file_format is None:
        file_format, param_file = resolve_format(infile)

    band = arguments["--band"]
    if band is None:
        band = 1
    else:
        band = int(band)

    supp_ndv = arguments["--ndv"]
    if supp_ndv is not None:
        supp_ndv = float(supp_ndv)
    
    if file_format == 'REAL4':
        data, driver, x, y, b, dtype = open_band_real4(infile, band, param_file, crop, cube=roicube)
    elif file_format == 'ROI_PAC':
        data, driver, x, y, b, dtype = open_band_roipac(infile, crop, supp_ndv)
    elif file_format == 'AMSTER':
        data, driver, x, y, b, dtype = open_band_amster(infile, param_file, crop)
    elif file_format == 'GAMMA':
        data, driver, x, y, b, dtype = open_band_gamma(infile, param_file, crop)
    elif file_format == 'GDAL':
        data, driver, x, y, b, dtype = open_band_gdal(infile, band, crop)

    display_raster_format(infile, driver, x, y, b, dtype)

    rad2mm = arguments["--rad2mm"]
    if rad2mm is not None:
        rad2mm = float(rad2mm)
    wrap = arguments["--wrap"]
    if wrap is not None:
        wrap = float(wrap)

    data[0] = correct_values_phase(data[0], ext, rad2mm, wrap, supp_ndv)
    if len(data) > 1:
        data[1] = correct_values_amp(data[1], ext)

    do_save = arguments["--save"]
    
    resolve_plot(data, arguments, crop)
    
    if do_save:
        plt.savefig(infile + '.pdf', format='PDF', dpi=180)
    plt.show()
    