# Absurd np.nan value conversion to actual NoData values
# Because np.nan is actually a value for the GeoTiff
# Inspiration from: https://mapscaping.com/nodata-raster-values-with-gdal/

# Usage: magic_cleaner_name.sh input_raster output_raster

na=$(gdalinfo $1 | grep "NoData Value" | sed 's/.*NoData Value=\([-0-9.eE+]\+\).*/\1/')
echo $na
gdal_calc.py -A $1 --outfile=$2 --calc="numpy.where(A == ""$na"", numpy.nan, A)"