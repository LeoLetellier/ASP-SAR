#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
std_map.py
__________

Usage: std_map.py <cube> --outfile=<o>
std_map.py -h | --help

Options:
  -h --help                 Show this screen
"""

import rioxarray as rxr
import dask.array as da
import docopt
    

def cli(arguments):
    data_cube = rxr.open_rasterio(
        arguments["<cube>"],
        masked=True,
    )
    # Ensure the data is chunked for dask
    if not isinstance(data_cube.data, da.Array):
        data_cube = data_cube.chunk({"band": -1, "x": 256, "y": 256})
    # Compute std along the band axis
    std_map_2d = data_cube.std(dim="band")
    std_map_2d.rio.to_raster(arguments["--outfile"])


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    cli(arguments)
