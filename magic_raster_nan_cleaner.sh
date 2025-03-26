# Absurd Nan Cleaner
# From: https://gis.stackexchange.com/questions/390438/replacing-nodata-values-by-a-constant-using-gdal-cli-tools
# does not impact the np.nan which is actually a value!

# Usage: magic_cleaner_name.sh input_raster output_raster new_nodata_value

na=$(gdalinfo $1 | grep "NoData Value" | sed 's/.*NoData Value=\([-0-9.eE+]\+\).*/\1/')
echo "NaN value in src: ""$na"
echo "Set NaN value to: $3"
gdalbuildvrt -srcnodata "$na" -vrtnodata 0 /vsistdout/ $1 | gdal_translate -a_nodata $3 /vsistdin/ $2