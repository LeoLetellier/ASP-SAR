#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cube_ts.py
---------------
Plot the time series of a GDAL raster cube (one band per date) at one or
more points, optionally relative to a reference point, averaging over a window.

Band descriptions in YYYYMMDD format are used as the band dates. If any band
description cannot be parsed, band numbers are used instead.

Usage:
  plot_cube_ts.py <cube> (<x> <y>)... [--ref=<x,y>] [--win=<n>] [--stat=<s>]
                  [--lonlat | --pix] [--nodata=<v>] [--vmin=<v>] [--vmax=<v>]
                  [--out=<file>] [--csv=<file>] [--noshow]
  plot_cube_ts.py -h | --help

Arguments:
  <cube>          GDAL raster cube (tif, vrt, ...), one band per date.
  <x> <y>         One or more points, in the cube CRS (default), lon/lat or pixel.
                  Repeat the pair for each point: cube.tif x1 y1 x2 y2 ...

Options:
  -h --help       Show this help.
  --ref=<x,y>     Reference point "x,y", same coordinate system as the points.
                  If given, its series is subtracted from every point series
                  and a second panel shows the differences.
                  Negative values need the "=" form: --ref=-118.1,34.2
  --win=<n>       Window size in pixels (odd, square) centred on each point [default: 1].
  --stat=<s>      Statistic over the window: mean or median [default: median].
  --lonlat        Interpret coordinates as lon,lat (EPSG:4326) and reproject to the cube CRS.
  --pix           Interpret coordinates as pixel col,row (0-based).
  --nodata=<v>    Override nodata value (default: the one stored in the file).
  --vmin=<v>      Lower y limit of the main plot.
  --vmax=<v>      Upper y limit of the main plot.
  --out=<file>    Save the figure to this file (png, pdf, ...).
  --csv=<file>    Export the time series to this CSV file.
  --noshow        Do not open an interactive window.

Note:
  Negative coordinates look like options to the parser. Put all options first
  and then use "--" before the positional arguments, e.g.
    plot_cube_ts.py --lonlat --win=3 -- cube.tif -118.25 34.05 -118.10 34.20
"""
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rasterio as rio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
import docopt


def to_float(text, name):
    try:
        return float(text)
    except ValueError:
        raise SystemExit(f"Could not parse {name}='{text}' as a number")


def parse_xy(text, name):
    """Parse 'a,b' into two floats."""
    parts = text.split(",")
    if len(parts) != 2:
        raise SystemExit(f"Could not parse {name}='{text}', expected 'x,y'")
    return to_float(parts[0], name), to_float(parts[1], name)


def to_pixel(src, x, y, lonlat=False, pix=False):
    """Convert input coordinates to (row, col) integer pixel indices."""
    if pix:
        return int(round(y)), int(round(x))
    if lonlat:
        if src.crs is None:
            raise SystemExit("Cube has no CRS, cannot use --lonlat")
        xs, ys = warp_transform("EPSG:4326", src.crs, [x], [y])
        x, y = xs[0], ys[0]
    row, col = src.index(x, y)
    return int(row), int(col)


def get_dates(src):
    """Return (x values, is_date flag) from band descriptions (YYYYMMDD)."""
    dates = []
    for desc in src.descriptions:
        try:
            dates.append(datetime.strptime(str(desc).strip(), "%Y%m%d"))
        except (ValueError, TypeError):
            print("Band descriptions are not all YYYYMMDD; using band numbers.")
            return np.arange(1, src.count + 1), False
    return np.array(dates), True


def window_series(src, row, col, win, stat, nodata):
    """Read all bands in a win x win window around (row, col) and reduce spatially."""
    half = win // 2
    if not (0 <= row < src.height and 0 <= col < src.width):
        raise SystemExit(f"Point (row={row}, col={col}) is outside the cube "
                         f"({src.height} rows x {src.width} cols)")

    # clip window to the raster extent
    r0, c0 = max(row - half, 0), max(col - half, 0)
    r1, c1 = min(row + half + 1, src.height), min(col + half + 1, src.width)
    data = src.read(window=Window(c0, r0, c1 - c0, r1 - r0)).astype("float64")

    nd = nodata if nodata is not None else src.nodata
    if nd is not None:
        data[data == nd] = np.nan
    data[~np.isfinite(data)] = np.nan

    flat = data.reshape(data.shape[0], -1)
    reducer = np.nanmedian if stat == "median" else np.nanmean
    with np.errstate(all="ignore"):
        return reducer(flat, axis=1)


def write_csv(path, x, is_date, names, series, ref_ts):
    """Write all time series to a wide-format CSV (one column per point)."""
    header = ["band", "date" if is_date else "index"]
    header += [f"{n}_value" for n in names]
    if ref_ts is not None:
        header += ["ref_value"] + [f"{n}_diff" for n in names]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, xi in enumerate(x):
            stamp = xi.strftime("%Y-%m-%d") if is_date else xi
            row = [i + 1, stamp] + [ts[i] for ts in series]
            if ref_ts is not None:
                row += [ref_ts[i]] + [ts[i] - ref_ts[i] for ts in series]
            w.writerow(row)
    print(f"Saved {path}")


def cli(arguments):
    cube = arguments["<cube>"]
    win = int(arguments["--win"])
    if win < 1 or win % 2 == 0:
        raise SystemExit("--win must be a positive odd integer")
    stat = arguments["--stat"]
    if stat not in ("mean", "median"):
        raise SystemExit("--stat must be 'mean' or 'median'")
    nodata = to_float(arguments["--nodata"], "--nodata") if arguments["--nodata"] else None
    vmin = to_float(arguments["--vmin"], "--vmin") if arguments["--vmin"] else None
    vmax = to_float(arguments["--vmax"], "--vmax") if arguments["--vmax"] else None
    lonlat, pix = arguments["--lonlat"], arguments["--pix"]

    xs_in = [to_float(v, "x") for v in arguments["<x>"]]
    ys_in = [to_float(v, "y") for v in arguments["<y>"]]

    names, series, pixels = [], [], []
    ref_ts, ref_pix = None, None

    with rio.open(cube) as src:
        print(f"{cube}: {src.width}x{src.height}, {src.count} bands, CRS={src.crs}")
        x, is_date = get_dates(src)

        for k, (px, py) in enumerate(zip(xs_in, ys_in), start=1):
            row, col = to_pixel(src, px, py, lonlat, pix)
            print(f"Point {k}    -> row={row}, col={col}")
            names.append(f"p{k}")
            pixels.append((row, col))
            series.append(window_series(src, row, col, win, stat, nodata))

        if arguments["--ref"]:
            rx, ry = parse_xy(arguments["--ref"], "--ref")
            ref_pix = to_pixel(src, rx, ry, lonlat, pix)
            print(f"Reference  -> row={ref_pix[0]}, col={ref_pix[1]}")
            ref_ts = window_series(src, *ref_pix, win, stat, nodata)

    if arguments["--csv"]:
        write_csv(arguments["--csv"], x, is_date, names, series, ref_ts)

    # ---- plotting ----
    nplots = 2 if ref_ts is not None else 1
    fig, axes = plt.subplots(nplots, 1, figsize=(10, 4 * nplots), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]

    for k, (name, (row, col), ts) in enumerate(zip(names, pixels, series)):
        color = f"C{k % 10}"
        axes[0].plot(x, ts, "o-", color=color,
                     label=f"{name} (r{row}, c{col}), {win}x{win} {stat}")
        if ref_ts is not None:
            axes[1].plot(x, ts - ref_ts, "o-", color=color, label=f"{name} - ref")

    if ref_ts is not None:
        axes[0].plot(x, ref_ts, "k--", marker="s", label=f"ref (r{ref_pix[0]}, c{ref_pix[1]})")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("Difference")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    axes[0].set_ylabel("Value")
    axes[0].set_title(cube)
    axes[0].set_ylim(bottom=vmin, top=vmax)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if is_date:
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
            mdates.AutoDateLocator()))
        axes[-1].set_xlabel("Date")
    else:
        axes[-1].set_xlabel("Band")

    fig.tight_layout()
    if arguments["--out"]:
        fig.savefig(arguments["--out"], dpi=150)
        print(f"Saved {arguments['--out']}")
    if not arguments["--noshow"]:
        plt.show()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    cli(arguments)