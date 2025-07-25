import numpy as np
from osgeo import gdal


def open_raster(path, band=1, crop=None, ndv_supp=None, get_ds=False):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    
    if crop is None:
        crop = [0, band.XSize, 0, band.YSize]
    
    data = band.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])
    
    data[data == ndv] = np.nan
    if ndv_supp is not None:
        data[data == ndv_supp] = np.nan

    if get_ds:
        return data, ds
    return data
    

def create_raster(path, data, offset=None, ndv=None, bands_nb=1):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(path, data.shape[1], data.shape[0], bands_nb, gdal.GDT_Int32)
    
    if offset is not None:
        ds.SetGeoTransform((offset(0), 1, 0, offset(1), 0, -1))

    band = ds.GetRasterBand(1)
    if ndv is not None:
        data[data == np.nan] = ndv
    band.WriteArray(data)

    ds.FlushCache()
    return ds


def create_copy_raster(path, template, data, crop=None, full=None, ndv=None):
    pass


