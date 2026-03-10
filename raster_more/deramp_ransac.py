#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deramp_ransac.py
________________

Use RANSAC solver to remove ramp to data on a raster.

This function can remove ramps with components along-track, across-track, in both dimensions, and using another raster data.
Rasters are subsampled for the ramp solving and opened chunk by chunk when saving data.

Usage: deramp_ransac.py <infile> --outfile=<outfile> [--ramp=<ramp>] [--add-data=<add-data>] [--plot] [--band=<band>] \
[--chunk-size=<chunk-size>] [--overwrite] [--ovr-ratio=<ovr-ratio>] [--ndv=<ndv>] [--mask-percentile=<mask-percentile>] \
[--save-ramp] [--nosave-deramp] [-v | --verbose] [--mask=<mask>] [-h | --help] [--save-coeffs] [--cyclic-data] \
[--weights=<weights>]
deramp_ransac.py -h | --help

Options:
  -h, --help           Show this screen
  <infile>              Raster to deramp
  --outfile=<outfile>   Path where the deramp data will be saved
  --ramp=<ramp>         Name of the ramp to use
  --add-data=<add-data>     Additionnal data file involved in the ramp
  --plot                Show QC plots after ramp solving
  --band=<band>         Band of the infile to use
  --chunk-size=<chunk-size> Size of the chunk to use when reading and writing data at full resolution. Can be one or two integers, comma separated
  --overwrite           Allow overwriting existing result
  --ovr-ratio=<ovr-ratio>   Ratio between 0 and 1 to downsample the data for ramp solving, or maximum numper of pixels if ratio > 1
  --ndv=<ndv>           Use an additionnal ndv when opening rasters
  --mask-percentile=<mask-percentile>   Mask that much extremum percentile from input raster, should be between 0 and 50 excluded
  --save-ramp           Additionnaly save the ramp itself at outfile_ramp.tif
  --nosave-deramp       Don't save the deramped data
  -v, --verbose         Show logging information
  --mask=<mask>         Use an additionnal custom mask to select pixels before ramp solving
  --save-coeffs         Output a text file with the ramp equation and associated coefficients
  --cyclic-data         Process the additionnal as a cyclic data in radians by using sin & cos decomposition
  --weights=<weights>   Add weights to use for the inversion

Ramps:
    - linear | s3
        linear (^1) in I and J [and D]
    - quadratic | s7
        quadratic [^1 and 2] in I and J [and D]
    - cubic
        quadratic [^1 and 2 and 3] in I and J [and D]
    - lin_range | s1
        linear in I [and D]
    - lin_azimuth | s2
        linear in J [and D]
    - lin_cross | s4
        linear in I, J and IJ [and D]
    - s5
        quadratic [^1 and 2] in I, linear in J [and D]
    - s6
        quadratic [^1 and 2] in J, linear in I [and D]
    - s8
        cubic [^1 and 2 and 3] in J, quadratic in I [^1 and 2] [and linear in D]
    - s9
        quadratic [^1 and 2] in IJ, linear in I and J [and D]
    - linear_data
        linear in D
    - quadratic_data
        quadratic [^1 and 2] in D
    - cubic_data
        cubic [^1 and 2 and 3] in D

Implementation: Leo Letellier
Ramp presets 's' from: [PyGdalSAR](https://github.com/simondaout/PyGdalSAR/blob/master/TimeSeries/invers_disp2coef.py)
"""

import docopt
import numpy as np
from osgeo import gdal
import os
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
import logging
from datetime import datetime


gdal.UseExceptions()
logger = logging.getLogger(__name__)


def discard_percentile(data: np.ndarray, percentile: float) -> np.ndarray:
    """Mask array percentiels

    Mask by applying nans for extremum values, based on percentile threshold.
    Will only keep values between [percentile ; 100 - percentile].

    Args:
        data (np.ndarray): Array to mask
        percentile (float): Percentile threshold to mask

    Returns:
        np.ndarray: Masked array
    """
    logger.info("Discarding based on percentile threshold: {}".format(percentile))
    if not 0 <= percentile < 50:
        raise ValueError(
            "Percentile {} should be between 0 and 50 both excluded".format(percentile)
        )
    data_max, data_min = np.nanpercentile(data, (100 - percentile, percentile))
    return np.where(np.logical_or(data < data_min, data > data_max), np.nan, data)


class Ramp:
    """Description of the components of the ramp, their order and associated coefficients

    A ramp is described by its components and associated orders, can then fit to data to
    determine coefficients and generate the corresponding raster ramp.

    Create the ramp with `Ramp.from_component_powers`

    Attributes:
        across_track_orders (list[int]): list of n powers associated to $i^n$, H / slant range
        along_track_orders (list[int]): list of n powers associated to $j^n$, V / azimuth
        cross_dimension_orders (list[int, int]): list of (n, m) powers associated to both image dimensions: $i^n.j^m$
        extern_data_orders (list[int]): list of n powers associated to a matching extern data raster
        coefficients (None | list[float]): proportionnality coefficient associated to each component for the solved ramp
    """

    def __init__(self):
        """Initialize an empty ramp"""
        # i^n
        self.__across_track_orders = []
        # j^n
        self.__along_track_orders = []
        # i^n.j^m
        self.__cross_dimension_orders = []
        # d^n
        self.__extern_data_orders = []

        # Resulting coefficients
        self.coefficients = None
        self.has_cyclic_extern_data = False

    @staticmethod
    def from_component_powers(
        across_track_orders: list[int] | None = None,
        along_track_orders: list[int] | None = None,
        cross_dimension_orders: list[int] | None = None,
        extern_data_orders: list[int] | None = None,
    ):
        ramp = Ramp()
        ramp.__across_track_orders = (
            across_track_orders if across_track_orders is not None else []
        )
        ramp.__along_track_orders = (
            along_track_orders if along_track_orders is not None else []
        )
        ramp.__cross_dimension_orders = (
            cross_dimension_orders if cross_dimension_orders is not None else []
        )
        ramp.__extern_data_orders = (
            extern_data_orders if extern_data_orders is not None else []
        )
        return ramp

    def __len__(self):
        add = 1  # reference
        if self.has_cyclic_extern_data:
            add += 1
        return (
            add
            + len(self.__across_track_orders)
            + len(self.__along_track_orders)
            + len(self.__cross_dimension_orders)
            + len(self.__extern_data_orders)
        )

    def __str__(self):
        equation = []
        coeff = 0

        for o in self.__across_track_orders:
            letter = chr(ord("a") + coeff)
            equation.append("{}.I^{}".format(letter, o))
            coeff += 1
        for o in self.__along_track_orders:
            letter = chr(ord("a") + coeff)
            equation.append("{}.J^{}".format(letter, o))
            coeff += 1
        for o in self.__cross_dimension_orders:
            letter = chr(ord("a") + coeff)
            equation.append("{}.I^{}.J^{}".format(letter, o[0], o[1]))
            coeff += 1
        if self.has_cyclic_extern_data:
            letter = chr(ord("a") + coeff)
            coeff += 1
            letter2 = chr(ord("a") + coeff)
            coeff += 1
            equation.append("{}.sin(D) + {}.cos(D)".format(letter, letter2))
        else:
            for o in self.__extern_data_orders:
                letter = chr(ord("a") + coeff)
                equation.append("{}.D^{}".format(letter, o))
                coeff += 1
        equation.append(chr(ord("a") + coeff))

        equation = "Ramp = " + " + ".join(equation)
        if self.coefficients is None or (
            len(self) != len(self.coefficients) and not self.has_cyclic_extern_data
        ):
            return equation

        coefficients = []
        for i, c in enumerate(self.coefficients):
            coefficients.append("{} = {}".format(chr(ord("a") + i), c))
        coefficients = "Coefficients" + "\n\t" + "\n\t".join(coefficients)

        return equation + "\n" + coefficients

    def get_matrix(
        self,
        across: np.ndarray,
        along: np.ndarray,
        extern_data: None | np.ndarray = None,
        with_reference: bool = True,
    ) -> np.ndarray:
        """Generate the matrix corresponding to the ramp components

        RANSAC solver already determines a reference so need no reference in this matrix,
        but it will be needed for ramp generation.
        """
        data_nb = len(across)
        if len(along) != data_nb:
            raise ValueError(
                "Size mismatch: across ({}) vs along ({})".format(data_nb, len(along))
            )
        if extern_data is not None and len(extern_data) != data_nb:
            raise ValueError(
                "Size mismatch: data_nb ({}) vs extern_data ({})".format(
                    data_nb, len(extern_data)
                )
            )
        if len(self.__extern_data_orders) > 0 and extern_data is None:
            raise ValueError(
                "Ramp got external data component but no external data is provided"
            )

        columns = []

        # Across track components
        for o in self.__across_track_orders:
            if o == 1:
                column = across
            elif o == 2:
                column = np.square(across)
            else:
                column = np.power(across, o)
            columns.append(column)

        # Along track components
        for o in self.__along_track_orders:
            if o == 1:
                column = along
            elif o == 2:
                column = np.square(along)
            else:
                column = np.power(along, o)
            columns.append(column)

        # Cross components
        for o in self.__cross_dimension_orders:
            column = np.power(across, o[0]) * np.power(along, o[1])
            columns.append(column)

        # Additional external data components
        for o in self.__extern_data_orders:
            if self.has_cyclic_extern_data and o == 1:
                # north = np.where(extern_data < 180, extern_data, 360 - extern_data)
                # east = extern_data - 90
                # east = np.where(east < 180, east, 360 - east)
                # columns.append(north)
                # columns.append(east)
                columns.append(np.sin(extern_data))
                columns.append(np.cos(extern_data))
            elif self.has_cyclic_extern_data:
                raise ValueError("Only linear for cyclic data")
            else:
                if o == 1:
                    column = extern_data
                elif o == 2:
                    column = np.square(extern_data)
                else:
                    column = np.power(extern_data, o)
            columns.append(column)

        # Constant reference
        if with_reference:
            columns.append(np.ones(shape=data_nb))

        return np.column_stack(columns)

    def solve(
        self,
        data: np.ndarray,
        extern_data: np.ndarray | None = None,
        mask_percentile: float = 0.5,
        weights: np.ndarray | None = None,
    ):
        """Solve the ramp using RANSAC and update coefficients"""
        data = discard_percentile(data, mask_percentile)

        if extern_data is not None:
            index = np.nonzero(~(np.isnan(data) | np.isnan(extern_data)))
            emi = extern_data[index].flatten()
        else:
            emi = None
            index = np.nonzero(~np.isnan(data))

        mi = data[index].flatten()
        logger.info("Solving ramp with {} samples".format(len(mi)))
        az = np.asarray(index[0]) / data.shape[0]
        rg = np.asarray(index[1]) / data.shape[1]

        G = self.get_matrix(rg, az, extern_data=emi, with_reference=False)

        if weights is not None:
            W = weights[index].flatten()
            G = W[:, np.newaxis] * G
            mi = W * mi

        ransac = RANSACRegressor(random_state=42)
        ransac.fit(G, mi)
        coeffs_list = ransac.estimator_.coef_.tolist()
        reference = float(ransac.estimator_.intercept_)
        coeffs_list.append(reference)

        self.coefficients = coeffs_list
        return self

    def generate(
        self,
        dimensions: list[int],
        chunk: list[int] | None = None,
        extern_data: np.ndarray | None = None,
    ):
        """Generate the ramp array possibly on a chunk of the array"""
        if len(self.coefficients) != len(self) and not self.has_cyclic_extern_data:
            raise AttributeError(
                "Number of coefficients ({}) in the ramp has change since last solving ({})".format(
                    len(self), len(self.coefficients)
                )
            )
        if chunk is None:
            xoff, yoff, xsize, ysize = 0, 0, dimensions[0], dimensions[1]
        else:
            xoff, yoff, xsize, ysize = chunk

        x_index = np.arange(xoff, xoff + xsize) / dimensions[0]
        y_index = np.arange(yoff, yoff + ysize) / dimensions[1]
        if extern_data is not None:
            extern_data = extern_data.flatten()

        along_values = np.repeat(y_index, xsize)
        across_values = np.tile(x_index, ysize)

        G = self.get_matrix(
            across_values, along_values, extern_data=extern_data, with_reference=True
        )

        ramp = np.dot(G, self.coefficients)
        ramp = ramp.reshape(ysize, xsize)

        return ramp


def gdal_raster_size(path: str) -> list[int]:
    """Get the raster size of the dataset from metadata only"""
    ds = gdal.Open(path)
    logger.info(
        "Raster {} has size {}".format(
            path, [ds.RasterXSize, ds.RasterYSize, ds.RasterCount]
        )
    )
    return ds.RasterXSize, ds.RasterYSize, ds.RasterCount


def open_gdal(
    path: str, band: int = 1, chunk: list[int] = None, add_ndv=None
) -> np.ndarray:
    """Open a raster possibly on a chunk"""
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(band)
    ndv = bd.GetNoDataValue()
    if chunk is not None:
        data = bd.ReadAsArray(chunk[0], chunk[1], chunk[2], chunk[3])
        logger.info("Opened raster {} with chunk {}".format(path, chunk))
    else:
        data = bd.ReadAsArray()
        logger.info("Opened full raster {}".format(path))
    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    if add_ndv is not None and add_ndv != np.nan:
        data[data == add_ndv] = np.nan
    return data


def open_gdal_reduced(
    path: str,
    band: int = 1,
    size: list[int] | None = None,
    resample_alg=gdal.GRIORA_Bilinear,
    add_ndv=None,
) -> np.ndarray:
    """Open the full raster with a given size and resampling algorithm"""
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(band)
    ndv = bd.GetNoDataValue()
    data = bd.ReadAsArray(
        buf_xsize=size[0], buf_ysize=size[1], resample_alg=resample_alg
    )
    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    if add_ndv is not None and add_ndv != np.nan:
        data[data == add_ndv] = np.nan
    logger.info(
        "Opened full raster {} with overview of size ({},{}) and resampling algorithm '{}'".format(
            path, size[0], size[1], resample_alg
        )
    )
    return data


def create_gdal_from_template(path: str, template: str):
    """Initialize a new raster with same config as an existing one (proj and geotransform) using gdal create copy"""
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName("GTiff").CreateCopy(path, ds_template)
    logger.info("Created raster {} from template {}".format(path, template))
    return ds


def write_gdal_chunk(
    path: str,
    data: np.ndarray,
    offset: list[int],
    band: int = 1,
    ndv: float | None = None,
):
    """Write a chunk into an existing raster"""
    ds = gdal.Open(path, gdal.GA_Update)
    band = ds.GetRasterBand(band)
    band_ndv = band.GetNoDataValue()

    if band_ndv is not None:
        ndv = band_ndv
    elif ndv is not None:
        band.SetNoDataValue(ndv)

    if ndv is not None and ndv != np.nan:
        data[data == np.nan] = ndv

    band.WriteArray(data, xoff=offset[0], yoff=offset[1])

    ds.FlushCache()


def define_chunks(dimensions: list[int], target_size: list[int]) -> list[int]:
    """Chunks are squared divisions of a grid each defined by [xoff, yoff, xsize, ysize]"""
    chunks = []
    # Number of full sized chunks
    full_x = dimensions[0] // target_size[0]
    full_y = dimensions[1] // target_size[1]
    # Residuals for padding chunks
    trunc_x = dimensions[0] % target_size[0]
    trunc_y = dimensions[1] % target_size[1]

    # Full sized chunks
    for kx in range(full_x):
        for ky in range(full_y):
            chunks.append(
                [
                    kx * target_size[0],
                    ky * target_size[1],
                    target_size[0],
                    target_size[1],
                ]
            )

    # Pad in x
    if trunc_x > 0:
        for ky in range(full_y):
            chunks.append(
                [full_x * target_size[0], ky * target_size[1], trunc_x, target_size[1]]
            )

    # Pad in y
    if trunc_y > 0:
        for kx in range(full_x):
            chunks.append(
                [kx * target_size[0], full_y * target_size[1], target_size[0], trunc_y]
            )

    # Pad the corner
    if trunc_x > 0 and trunc_y > 0:
        chunks.append(
            [full_x * target_size[0], full_y * target_size[1], trunc_x, trunc_y]
        )

    logger.info("Decomposed the raster dimensions into {} chunks".format(len(chunks)))

    return chunks


def plot_raster_bar(fig, ax, raster: np.ndarray, label: str, vminmax):
    """Plot array data with a colorbar on a specific axis and figure"""
    im = ax.imshow(raster, cmap="RdBu_r", vmin=-vminmax, vmax=+vminmax)
    ax.set_title(label)
    fig.colorbar(im, ax=ax, fraction=0.07, pad=0.04)


def plot_ramp_result(data: np.ndarray, ramp: np.ndarray):
    """QC plot of the original data, the estimated ramp and the deramp data"""
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    vminmax = max(
        abs(np.nanpercentile(data - ramp, 2)), abs(np.nanpercentile(data - ramp, 98))
    )
    plot_raster_bar(fig, axs[0], data, "data", vminmax)
    plot_raster_bar(fig, axs[1], ramp, "ramp", vminmax)
    plot_raster_bar(fig, axs[2], data - ramp, "deramp", vminmax)

    plt.tight_layout()


def get_ramp(
    ramp_name: str, has_extern_data: bool = False, use_cyclic_data: bool = False
) -> Ramp:
    """Create a ramp based on a ramp name"""
    across_track_orders = None
    extern_data_orders = None
    along_track_orders = None
    cross_dimensions_order = None

    if ramp_name == "linear" or ramp_name == "s3":
        across_track_orders = [1]
        along_track_orders = [1]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "quadratic" or ramp_name == "s7":
        across_track_orders = [1, 2]
        along_track_orders = [1, 2]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "cubic":
        across_track_orders = [1, 2, 3]
        along_track_orders = [1, 2, 3]
        if has_extern_data:
            extern_data_orders = [1, 2, 3]
    elif ramp_name == "lin_range" or ramp_name == "s1":
        across_track_orders = [1]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "lin_azimuth" or ramp_name == "s2":
        along_track_orders = [1]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "lin_cross" or ramp_name == "s4":
        across_track_orders = [1]
        along_track_orders = [1]
        cross_dimensions_order = [[1, 1]]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "s5":
        across_track_orders = [2, 1]
        along_track_orders = [1]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "s6":
        across_track_orders = [1]
        along_track_orders = [2, 1]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "s8":
        across_track_orders = [3, 2, 1]
        along_track_orders = [2, 1]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "s9":
        across_track_orders = [1]
        along_track_orders = [1]
        cross_dimensions_order = [[1, 2]]
        if has_extern_data:
            extern_data_orders = [1]
    elif ramp_name == "linear_data" and has_extern_data:
        extern_data_orders = [1]
    elif ramp_name == "quadratic_data" and has_extern_data:
        extern_data_orders = [2, 1]
    elif ramp_name == "cubic_data" and has_extern_data:
        extern_data_orders = [3, 2, 1]
    else:
        # could add expression parsing here
        raise ValueError("Ramp preset does not exists")

    ramp = Ramp.from_component_powers(
        across_track_orders=across_track_orders,
        along_track_orders=along_track_orders,
        cross_dimension_orders=cross_dimensions_order,
        extern_data_orders=extern_data_orders,
    )
    ramp.has_cyclic_extern_data = use_cyclic_data
    return ramp


def check_raster_consistency(*files):
    raster_size = gdal_raster_size(files[0])
    for f in files:
        if f is not None:
            f_size = gdal_raster_size(f)
            # Dont check the band nb
            if f_size[:2] != raster_size[:2]:
                raise ValueError(
                    "Raster sizes are inconsistents, expected {} but got {} for file {}".format(
                        raster_size, f_size, f
                    )
                )
    return raster_size


def infer_ovr_size(ratio, raster_size):
    if ratio > 1:
        # use maximum number of pixel instead
        ratio = np.sqrt(ratio / (raster_size[0] * raster_size[1]))
        if ratio <= 0:
            raise ValueError("Ratio is not achievable")
        elif ratio > 1:
            ratio = 1
    size = [int(ratio * raster_size[0]), int(ratio * raster_size[1])]
    return size


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    if arguments["--verbose"]:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s: %(asctime)s | %(message)s"
        )
    infile = arguments["<infile>"]
    outfile = arguments["--outfile"]
    ramp_name = arguments["--ramp"]
    do_plot = arguments["--plot"]
    band = int(arguments["--band"]) if arguments["--band"] is not None else 1
    target_size = (
        [int(s) for s in arguments["--chunk-size"].split(",")]
        if arguments["--chunk-size"] is not None
        else [1024, 1024]
    )
    if len(target_size) == 1:
        target_size = [target_size[0], target_size[0]]
    elif len(target_size) > 2:
        raise ValueError(
            "chunk size have max 2 values but {} were provided".format(len(target_size))
        )
    overwrite = arguments["--overwrite"]
    ratio = (
        float(arguments["--ovr-ratio"]) if arguments["--ovr-ratio"] is not None else 1e6
    )
    add_ndv = float(arguments["--ndv"]) if arguments["--ndv"] is not None else None
    mask_percentile = (
        arguments["--mask-percentile"]
        if arguments["--mask-percentile"] is not None
        else 0.2
    )
    add_data = arguments["--add-data"]
    do_save_ramp = arguments["--save-ramp"]
    do_save_deramp = not arguments["--nosave-deramp"]
    if do_save_ramp:
        outfile_ramp = outfile + "_ramp.tif"
    mask = arguments["--mask"]
    save_coeffs = arguments["--save-coeffs"]
    use_cyclic_data = arguments["--cyclic-data"]
    weights = arguments["--weights"]

    # Create and solve the ramp
    ramp = get_ramp(
        ramp_name, has_extern_data=not add_data is None, use_cyclic_data=use_cyclic_data
    )
    raster_size = check_raster_consistency(infile, add_data, mask, weights)
    ovr_size = infer_ovr_size(ratio, raster_size)
    data_reduced = open_gdal_reduced(infile, band=band, size=ovr_size, add_ndv=add_ndv)
    if mask is not None:
        mask_reduced = open_gdal_reduced(mask, size=ovr_size, add_ndv=add_ndv)
        data_reduced = np.where(mask_reduced >= 1, data_reduced, np.nan)
    if weights is not None:
        weights_reduced = open_gdal_reduced(weights, size=ovr_size, add_ndv=add_ndv)
    else:
        weights_reduced = None
    if add_data is not None:
        add_data_reduced = open_gdal_reduced(add_data, size=ovr_size, add_ndv=add_ndv)
        ramp.solve(data_reduced, add_data_reduced, mask_percentile, weights_reduced)
    else:
        ramp.solve(data_reduced, None, mask_percentile, weights_reduced)
    if save_coeffs:
        outfile_coeffs = outfile + "_coeffs_{}.txt".format(
            datetime.now().strftime("%y%m%d%H%M%S")
        )
        logger.info("Saving coefficients to {}".format(outfile_coeffs))
        with open(outfile_coeffs, "w") as f:
            f.write(str(ramp))

    if do_plot:
        # QC plots
        import matplotlib.pyplot as plt

        logger.info("Solved ramp:\n{}".format(ramp))

        if add_data is not None:
            ramp_array = ramp.generate(ovr_size, extern_data=add_data_reduced)
        else:
            ramp_array = ramp.generate(ovr_size)

        plot_ramp_result(data_reduced, ramp_array)
        plt.show()

    # Don't need the reduced overview anymore
    del data_reduced
    if add_data is not None:
        del add_data_reduced

    # Saving outputs
    if do_save_deramp or do_save_ramp:
        # Prepare new raster files with matching projection and geotransform
        if os.path.isfile(outfile):
            if not overwrite:
                raise FileExistsError(
                    "Raster already exists: {}. Use --overwrite".format(outfile)
                )
        else:
            create_gdal_from_template(outfile, infile)

        if do_save_ramp:
            if os.path.isfile(outfile_ramp):
                if not overwrite:
                    raise FileExistsError(
                        "Raster already exists: {}. Use --overwrite".format(
                            outfile_ramp
                        )
                    )
            else:
                create_gdal_from_template(outfile_ramp, infile)

        # Divide into chunk for the full resolution ramp generation, correction and saving
        chunks = define_chunks([raster_size[0], raster_size[1]], target_size)
        for i, c in enumerate(tqdm(chunks, unit="chunk")):
            data = open_gdal(infile, band, chunk=c, add_ndv=add_ndv)
            if add_data is not None:
                add_data_chunk = open_gdal(add_data, chunk=c, add_ndv=add_ndv)
                ramp_array = ramp.generate(
                    raster_size, chunk=c, extern_data=add_data_chunk
                )
            else:
                ramp_array = ramp.generate(raster_size, chunk=c)

            # Chunk defines the x and y offsets at [0] and [1] positions
            if do_save_ramp:
                write_gdal_chunk(outfile_ramp, ramp_array, [c[0], c[1]])
            if do_save_deramp:
                write_gdal_chunk(outfile, data - ramp_array, [c[0], c[1]])
