#!/usr/bin/env python3
"""
extract_timeseries.py

Extract a time series from an ENVI (GDAL-readable) multitemporal cube for
an arbitrary list of points. For each point, a window of size
(2*half_window+1) x (2*half_window+1) centered on the point is read per
band, and the NaN-ignoring median is computed.

Band descriptions are expected to encode the acquisition date as YYYYMMDD.

Usage:
    python extract_timeseries.py \
        --cube path/to/cube \
        --points 120,340 150,300 80,222 \
        --window 1 \
        --output timeseries.txt \
        --nodata -9999

    # or read points from a file (one "col,row" or "label,col,row" per line):
    python extract_timeseries.py \
        --cube path/to/cube \
        --points-file points.csv \
        --window 1 \
        --output timeseries.txt

Notes:
    Each point is given as: col,row  (x,y in pixel/image coordinates, 0-based)
    Optionally prefix with a label: label,col,row
    --window is the half-window size w, so the window is (2w+1) x (2w+1).
"""

import argparse
import csv
import re
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True, help="Path to the ENVI cube (.dat/.img/.bin, etc.)")
    p.add_argument("--points", nargs="+", metavar="COL,ROW or LABEL,COL,ROW",
                    help="List of points, e.g. --points 120,340 150,300")
    p.add_argument("--points-file",
                    help="CSV/text file with one point per line: col,row or label,col,row")
    p.add_argument("--window", type=int, default=1,
                    help="Half-window size w (window is (2w+1)x(2w+1)). Default: 1")
    p.add_argument("--output", required=True, help="Output text file path")
    p.add_argument("--nodata", type=float, default=None,
                    help="Optional nodata value to mask out (in addition to NaN) before "
                         "computing the median")
    p.add_argument("--sep", default="\t", help="Field separator in output file. Default: tab")
    args = p.parse_args()

    if not args.points and not args.points_file:
        p.error("Provide points via --points or --points-file")

    return args


def parse_point_token(token, default_index):
    """
    Parse a single point token of the form 'col,row' or 'label,col,row'.
    Returns (label, col, row).
    """
    parts = [t.strip() for t in token.split(",") if t.strip() != ""]
    if len(parts) == 2:
        col, row = int(parts[0]), int(parts[1])
        label = f"pt{default_index}"
    elif len(parts) == 3:
        label = parts[0]
        col, row = int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Cannot parse point '{token}'. Expected 'col,row' or 'label,col,row'.")
    return label, col, row


def load_points(args):
    points = []  # list of (label, col, row)
    idx = 1

    if args.points:
        for token in args.points:
            label, col, row = parse_point_token(token, idx)
            points.append((label, col, row))
            idx += 1

    if args.points_file:
        with open(args.points_file, newline="") as f:
            reader = csv.reader(f)
            for row_data in reader:
                if not row_data or row_data[0].strip().startswith("#"):
                    continue
                token = ",".join(row_data)
                label, col, row = parse_point_token(token, idx)
                points.append((label, col, row))
                idx += 1

    if not points:
        sys.exit("ERROR: no points parsed from --points / --points-file")

    return points


def extract_date_from_description(desc, band_index):
    """
    Extract a YYYYMMDD date string from a band description.
    Falls back to a placeholder if no date pattern is found.
    """
    if desc:
        match = re.search(r"(19|20)\d{6}", desc)
        if match:
            return match.group(0)
    return f"band_{band_index}"


def read_window_median(band, col, row, half_window, xsize, ysize, nodata=None):
    """
    Read a (2*half_window+1) x (2*half_window+1) window centered on
    (col, row) from a GDAL band, clipped to raster bounds, and return
    the NaN-ignoring median. Also masks the given nodata value if provided.
    """
    x_off = max(col - half_window, 0)
    y_off = max(row - half_window, 0)
    x_end = min(col + half_window + 1, xsize)
    y_end = min(row + half_window + 1, ysize)

    win_xsize = x_end - x_off
    win_ysize = y_end - y_off

    if win_xsize <= 0 or win_ysize <= 0:
        return np.nan

    arr = band.ReadAsArray(x_off, y_off, win_xsize, win_ysize).astype(np.float64)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    band_nodata = band.GetNoDataValue()
    if band_nodata is not None:
        arr[arr == band_nodata] = np.nan

    if np.all(np.isnan(arr)):
        return np.nan

    return np.nanmedian(arr)


def main():
    args = parse_args()
    points = load_points(args)  # list of (label, col, row)

    ds = gdal.Open(args.cube, gdal.GA_ReadOnly)
    if ds is None:
        sys.exit(f"ERROR: could not open cube: {args.cube}")

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    n_bands = ds.RasterCount

    for label, col, row in points:
        if not (0 <= col < xsize and 0 <= row < ysize):
            sys.exit(f"ERROR: point '{label}' ({col}, {row}) is outside raster extent "
                      f"({xsize} x {ysize})")

    # results[band_index] = (date, [val_pt1, val_pt2, ...])
    results = []

    for b in range(1, n_bands + 1):
        band = ds.GetRasterBand(b)
        desc = band.GetDescription()
        date = extract_date_from_description(desc, b)

        values = [
            read_window_median(band, col, row, args.window, xsize, ysize, args.nodata)
            for (_, col, row) in points
        ]
        results.append((date, values))

    def sort_key(item):
        d = item[0]
        return (0, d) if d.isdigit() and len(d) == 8 else (1, d)

    results.sort(key=sort_key)

    # Build header with coordinates for each point, e.g. "P1[col=120,row=340]"
    col_headers = ["date"] + [
        f"{label}[col={col},row={row}]" for (label, col, row) in points
    ]

    with open(args.output, "w") as f:
        f.write(args.sep.join(col_headers) + "\n")
        for date, values in results:
            val_strs = ["NaN" if np.isnan(v) else f"{v:.6f}" for v in values]
            f.write(args.sep.join([date] + val_strs) + "\n")

    print(f"Wrote {len(results)} records for {len(points)} points to {args.output}")

    ds = None


if __name__ == "__main__":
    main()