#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rfft.py
-------------
Compute a fft 2d for a GDAL raster

Usage: rfft.py <infile> [--band=<band> --ndv=<ndv>] [--dx=<dx>] [--dy=<dy>] [--cut=<cut>]


Options:
-h --help               Show this screen.
<infile>                Raster to be displayed
"""

import docopt
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
gdal.UseExceptions()
from skimage.filters import butterworth


def open_gdal(file, band, supp_ndv=None):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    data = band.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        data[data==ndv] = np.nan
    if supp_ndv is not None and supp_ndv != np.nan:
        data[data==supp_ndv] = np.nan
    return data


def fft_spectrum(data):
    data = np.nan_to_num(data, nan=np.nanmean(data))
    fft = fft2(data)
    power_spectrum = np.abs(fft) ** 2
    return fftshift(power_spectrum)


def plot_spectrum(data, dx=1, dy=1):
    spectrum = fft_spectrum(data)
    fx = fftshift(np.fft.fftfreq(data.shape[1], dx))
    fy = fftshift(np.fft.fftfreq(data.shape[0], dy))
    plt.figure()
    plt.imshow(np.log1p(spectrum), cmap='magma', extent=[
        fx.min(), fx.max(),
        fy.min(), fy.max(),
    ])
    plt.colorbar()
    plt.title("Frequency Spectrum (FFT2)")


def roll3_wrap(data):
    data = data + 2 / 3 * np.pi
    data[data > np.pi] -= 2 * np.pi
    return data


def wrap_red_but(data, cut, highpass=False):
    data = np.nan_to_num(data, nan=np.nanmean(data))
    buts = []
    for _ in range(3):
        but = butterworth(data, cut, high_pass=highpass)
        data = roll3_wrap(data)
        buts.append(but)
    but_reduced = np.median(buts, axis=0)
    return but_reduced


def plot_data_freq(data, cut, dx=1, dy=1, highpass=False):
    data = np.nan_to_num(data, nan=np.nanmean(data))

    fs = 1 / np.sqrt(dx**2 + dy**2)

    phasor = phasor_filter(data, cut, highpass=highpass)
    return phasor

    but = butterworth(data, cut, high_pass=highpass)
    return but


    fft = fftshift(fft2(data))
    fx = fftshift(np.fft.fftfreq(data.shape[1], dx))
    fy = fftshift(np.fft.fftfreq(data.shape[0], dy))
    FX, FY = np.meshgrid(fy, fx, indexing='ij')
    R = np.sqrt(FX**2 + FY**2)
    if lowpass:
        mask = R <= cut
    else:
        mask = R > cut
    data_cut = np.real(ifft2(fftshift(fft * mask)))
    return data_cut


def plot_img(data, cmap):
    plt.figure()
    plt.imshow(data, cmap=cmap, vmin=-np.pi, vmax=np.pi)

def find_variation_cutoff(data, dx=1, dy=1, threshold=np.pi/2):
    """
    Find the spatial frequency cutoff (normalized, 0-0.5) below which
    the cumulative spectral energy of the gradient magnitude reaches
    the threshold variation level.
    
    Returns the normalized frequency to pass as --cut.
    """
    data_clean = np.nan_to_num(data, nan=np.nanmean(data))

    # Gradient magnitude = variation field
    gy, gx = np.gradient(data_clean, dy, dx)
    variation = np.sqrt(gx**2 + gy**2)

    # FFT of the variation field
    fft_var = fftshift(fft2(variation))
    power = np.abs(fft_var) ** 2

    # Radial frequency axis (normalized, 0-0.5)
    fy = fftshift(np.fft.fftfreq(data.shape[0]))  # dx/dy cancel in normalization
    fx = fftshift(np.fft.fftfreq(data.shape[1]))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)

    # Radial profile: bin power by frequency radius
    r_bins = np.linspace(0, 0.5, 256)
    radial_power = np.array([
        power[(R >= r_bins[i]) & (R < r_bins[i+1])].sum()
        for i in range(len(r_bins) - 1)
    ])
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Cumulative power, normalized to [0, 1]
    cum_power = np.cumsum(radial_power)
    cum_power /= cum_power[-1]

    # Find the mean variation level at each radial band
    # and locate where it first exceeds pi/2
    radial_mean_var = np.array([
        variation[(R >= r_bins[i]) & (R < r_bins[i+1])].mean()
        if (R >= r_bins[i]).any() else 0
        for i in range(len(r_bins) - 1)
    ])

    # Cutoff = lowest frequency whose band mean variation exceeds threshold
    candidates = r_centers[radial_mean_var > threshold]
    if len(candidates) == 0:
        print("Warning: no frequency band exceeds the threshold — data variation is globally low.")
        return None

    cutoff = candidates[0]
    print(f"Suggested --cut={cutoff:.4f}  (normalized freq, 0-0.5 range)")

    # # Optional diagnostic plot
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # ax1.plot(r_centers, radial_mean_var)
    # ax1.axhline(threshold, color='r', linestyle='--', label='π/2 threshold')
    # ax1.axvline(cutoff, color='orange', linestyle='--', label=f'cut={cutoff:.4f}')
    # ax1.set_ylabel("Mean gradient magnitude")
    # ax1.legend()
    # ax2.plot(r_centers, cum_power)
    # ax2.axvline(cutoff, color='orange', linestyle='--')
    # ax2.set_ylabel("Cumulative power (normalized)")
    # ax2.set_xlabel("Normalized spatial frequency")
    # plt.tight_layout()

    return cutoff


def phasor_filter(data, cut, highpass=False):
    """
    Filter wrapped phase via its complex phasor to avoid
    treating wrap jumps as real high-frequency signal.
    
    data : wrapped phase (radians, any range)
    cut  : normalized frequency cutoff (0-0.5)
    """
    # Convert to unit phasor — wrapping discontinuities disappear
    phasor = np.exp(1j * data)

    # Filter real and imaginary parts independently
    # (they are both smooth continuous signals)
    real_filt = butterworth(phasor.real, cut, high_pass=highpass)
    imag_filt = butterworth(phasor.imag, cut, high_pass=highpass)

    # Reconstruct phase from filtered phasor
    phasor_filt = real_filt + 1j * imag_filt
    return np.angle(phasor_filt)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["<infile>"]
    band = arguments["--band"]
    band = 1 if band is None else int(band)
    ndv = arguments["--ndv"]
    ndv = None if ndv is None else float(ndv)
    dx = arguments["--dx"]
    dy = arguments["--dy"]
    dx = 1 if dx is None else float(dx)
    dy = 1 if dy is None else float(dy)
    cut = arguments["--cut"]
    cut = None if cut is None else float(cut)

    data = open_gdal(infile, band, ndv)

    cmap = "RdBu_r"
    if data.dtype == np.complex64:
        data = np.angle(data)
        cmap = "twilight"

    plot_spectrum(data, dx, dy)

    # if cut is not None:
    #     print(cut)
    #     cut = find_variation_cutoff(data, dx, dy, threshold=cut)
    #     print(cut)

    data_low = plot_data_freq(data, cut, dx, dy, highpass=False)
    # data_high = plot_data_freq(data, cut, dx, dy, highpass=True)
    data_high = data - data_low

    # data_low = wrap_red_but(data, cut, highpass=False)
    # data_high = wrap_red_but(data, cut, highpass=True)

    plot_img(data_low, cmap)
    plt.title("data low pass")
    plot_img(data_high, cmap)
    plt.title("data high pass")
    plot_img(data_low + data_high, cmap)
    plt.title("low+high")
    gx, gy = np.gradient(data_high)
    plot_img(np.sqrt(gx**2+gy**2), "Greys_r")
    plt.title("grad")

    plt.figure()
    p2, p98 = np.nanpercentile(data_high.flatten(), (2, 98))
    plt.hist(data_high.flatten(), bins=30, range=(p2, p98))

    plt.show()
