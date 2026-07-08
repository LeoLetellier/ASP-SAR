#!/usr/bin/env python3
"""
geotiff_to_gif.py

Convert a series of GeoTIFF files into an animated GIF with:
- consistent min/max scaling across all frames
- a colormap applied to single-band data
- a text label burned into each frame (filename by default)

Usage:
    python geotiff_to_gif.py "data/*.tif" -o output.gif
    python geotiff_to_gif.py "data/*.tif" -o output.gif --cmap terrain --fps 8
    python geotiff_to_gif.py "data/*.tif" --vmin 0 --vmax 6000 --band 1
    python geotiff_to_gif.py "data/*.tif" --no-label
"""

import argparse
import glob
import os
import sys

import numpy as np
import rasterio
import imageio.v2 as imageio
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser(description="Convert a series of GeoTIFFs into an animated GIF.")
    p.add_argument("pattern", help="Wildcard pattern for input GeoTIFFs, e.g. 'data/*.tif' (quote it so the shell doesn't expand it)")
    p.add_argument("-o", "--output", default="output.gif", help="Output GIF path (default: output.gif)")
    p.add_argument("--band", type=int, default=1, help="Band index to read, 1-based (default: 1)")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap name (default: viridis)")
    p.add_argument("--vmin", type=float, default=None, help="Fixed minimum value for scaling (default: computed from data)")
    p.add_argument("--vmax", type=float, default=None, help="Fixed maximum value for scaling (default: computed from data)")
    p.add_argument("--fps", type=float, default=5, help="Frames per second (default: 5)")
    p.add_argument("--loop", type=int, default=0, help="Loop count, 0 = infinite (default: 0)")
    p.add_argument("--downsample", type=int, default=1, help="Integer downsampling factor, e.g. 2 = half resolution (default: 1)")
    p.add_argument("--no-label", action="store_true", help="Disable the on-frame text label")
    p.add_argument("--label-source", choices=["filename", "index"], default="filename",
                   help="What to print as the label (default: filename)")
    p.add_argument("--nodata-color", default="255,255,255", help="RGB for nodata pixels, comma-separated (default: 255,255,255)")
    return p.parse_args()


def find_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files matched pattern: {pattern}")
    print(f"Found {len(files)} files")
    return files


def compute_global_range(files, band, downsample, vmin_arg, vmax_arg):
    if vmin_arg is not None and vmax_arg is not None:
        return vmin_arg, vmax_arg

    vmin, vmax = np.inf, -np.inf
    for f in files:
        with rasterio.open(f) as src:
            out_shape = (src.height // downsample, src.width // downsample) if downsample > 1 else None
            arr = src.read(band, out_shape=out_shape).astype(float)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            if np.all(np.isnan(arr)):
                continue
            vmin = min(vmin, np.nanmin(arr))
            vmax = max(vmax, np.nanmax(arr))

    vmin = vmin_arg if vmin_arg is not None else vmin
    vmax = vmax_arg if vmax_arg is not None else vmax

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        sys.exit("Could not determine a valid data range. Check --band, --vmin, --vmax, or your nodata values.")

    print(f"Global data range: {vmin:.4g} to {vmax:.4g}")
    return vmin, vmax


def build_frame(f, index, band, downsample, vmin, vmax, colormap, nodata_rgb, add_label, label_source, font):
    with rasterio.open(f) as src:
        out_shape = (src.height // downsample, src.width // downsample) if downsample > 1 else None
        arr = src.read(band, out_shape=out_shape).astype(float)
        mask = np.zeros(arr.shape, dtype=bool)
        if src.nodata is not None:
            mask = arr == src.nodata
            arr = np.where(mask, np.nan, arr)

    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0, 1)

    rgba = colormap(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[mask] = nodata_rgb

    img = Image.fromarray(rgb)

    if add_label:
        label = os.path.splitext(os.path.basename(f))[0] if label_source == "filename" else str(index)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 6
        draw.rectangle([4, 4, 4 + text_w + pad * 2, 4 + text_h + pad * 2], fill=(0, 0, 0))
        draw.text((4 + pad, 4 + pad - bbox[1]), label, fill=(255, 255, 255), font=font)

    return np.array(img)


def main():
    args = parse_args()
    files = find_files(args.pattern)

    try:
        nodata_rgb = tuple(int(x) for x in args.nodata_color.split(","))
        assert len(nodata_rgb) == 3
    except Exception:
        sys.exit("--nodata-color must be three comma-separated integers, e.g. '255,255,255'")

    try:
        colormap = cm.get_cmap(args.cmap)
    except ValueError:
        sys.exit(f"Unknown colormap '{args.cmap}'. See https://matplotlib.org/stable/users/explain/colors/colormaps.html")

    vmin, vmax = compute_global_range(files, args.band, args.downsample, args.vmin, args.vmax)

    font = None if args.no_label else ImageFont.load_default()

    frames = []
    for i, f in enumerate(files):
        frame = build_frame(
            f, i, args.band, args.downsample, vmin, vmax, colormap,
            nodata_rgb, add_label=not args.no_label, label_source=args.label_source, font=font,
        )
        frames.append(frame)
        print(f"  processed {i + 1}/{len(files)}: {os.path.basename(f)}")

    imageio.mimsave(args.output, frames, fps=args.fps, loop=args.loop)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()