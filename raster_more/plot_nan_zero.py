#!/usr/bin/env python3
"""plot_nan_zero.py — open a raster with GDAL and show where NaNs and zeros are."""
import argparse
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    ap = argparse.ArgumentParser(description="Plot NaN and zero locations in a raster")
    ap.add_argument('file', help='Path to raster (GeoTIFF, ENVI, etc.)')
    ap.add_argument('--band', type=int, default=1, help='Band number (default 1)')
    args = ap.parse_args()

    ds = gdal.Open(args.file)
    if ds is None:
        raise RuntimeError(f"GDAL could not open {args.file}")

    band = ds.GetRasterBand(args.band)
    arr = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()

    print(f"Driver:  {ds.GetDriver().ShortName}")
    print(f"Size:    {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"NoData:  {nodata}")

    nan_mask = np.isnan(arr)
    zero_mask = (arr == 0)

    total = arr.size
    print(f"NaNs:  {nan_mask.sum()} ({100*nan_mask.sum()/total:.2f}%)")
    print(f"Zeros: {zero_mask.sum()} ({100*zero_mask.sum()/total:.2f}%)")

    # ---- combined categorical map: 0=valid, 1=zero, 2=nan ----
    combo = np.zeros(arr.shape, dtype=np.uint8)
    combo[zero_mask] = 1
    combo[nan_mask] = 2

    cmap = mcolors.ListedColormap(['lightgray', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(zero_mask, cmap='Reds', interpolation='none')
    axes[0].set_title(f'Zeros ({zero_mask.sum()} px)')
    axes[0].axis('off')

    axes[1].imshow(nan_mask, cmap='Blues', interpolation='none')
    axes[1].set_title(f'NaNs ({nan_mask.sum()} px)')
    axes[1].axis('off')

    im = axes[2].imshow(combo, cmap=cmap, norm=norm, interpolation='none')
    axes[2].set_title('Combined (gray=valid, red=zero, blue=nan)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('nan_zero_plot.png', dpi=150)
    plt.show()
    print("Saved: nan_zero_plot.png")


if __name__ == '__main__':
    main()