#!/usr/bin/env python3
"""Resample a raster onto the exact grid of a reference raster."""

import argparse

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def match_grid(input_path, reference_path, output_path, method="bilinear", dst_nodata=None):
    with rasterio.open(reference_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()

        nodata = dst_nodata if dst_nodata is not None else src.nodata
        dtype = src.dtypes[0]

        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            nodata=nodata,
            compress="deflate",
            tiled=True,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                dest = np.full(
                    (dst_height, dst_width),
                    nodata if nodata is not None else 0,
                    dtype=dtype,
                )
                reproject(
                    source=rasterio.band(src, band),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=nodata,
                    resampling=Resampling[method],
                )
                dst.write(dest, band)

    print(f"Written: {output_path} ({dst_width}x{dst_height}, {dst_crs})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Raster to resample")
    p.add_argument("reference", help="Raster whose grid should be matched")
    p.add_argument("output", help="Output GeoTIFF")
    p.add_argument(
        "-r", "--method",
        default="bilinear",
        choices=[m.name for m in Resampling if m.value <= 7 or m.name in ("average", "mode")],
        help="Resampling method (default: bilinear). Use 'nearest' for categorical data.",
    )
    p.add_argument("--nodata", type=float, default=None, help="Override output NoData value")
    args = p.parse_args()

    match_grid(args.input, args.reference, args.output, args.method, args.nodata)