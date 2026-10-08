# SAR Image Correlation Workflow


## Preparing the SAR images: AMSTer ToolBox

### Reading the SLC images

SAR images are red using the AMSTer ToolBox, please refeer to their [documentation](https://github.com/AMSTerUsers/AMSTer_Distribution) for exact usage.

All images must be converted in `csl`, the internal AMSTer format. For example, using S1:

```shell
Read_all_img.sh SAFE_PATH DST S1 KML_PATH VV EXACTKML ForceAllYears
```

Images are converted from their SAFE folder into the AMSTer CSL folder.

The images are then coregistrated using a DEM against a primary reference acquisition on a common grid / geometry:

```shell
ALL2GIF.sh PRIMARY_DATE PARAMETER_FILE 100 100
```

Coregistrated images are stored in the SM folder.

After coregistration, update internal paths in these folders by running:

```shell
RenamePathAfterMove_in_SAR_SM_AMPLITUDES.sh S1
```

## Preparing for Correlation

Transfer and preprocess the coregistrated images in a dedicated working directory:

```shell
amster2aspsar.py /full-path/AMSTer/SAR_SM/AMPLITUDES/SAT/TRK/REGION /ASP-SAR/working-dir --s1
```

Images are converted in decibel and stored as GTiff.

**Optionnal**: Apply temporal stacking on the images by adding --stack-bt=50 (moving median stacking over 50 days) to the previous command.

**Optionnal**: Apply a mask based on a DA threshold and clean outliers in the `STACKTIF` folder:

```shell
find 20*_stack50.tif -maxdepth 0 -exec r_ps_select.py --infile={} --outfile={}_da_mask.tif --da=../GEOTIFF/AMPLI_DA.tif --threshold=0.12 \;
find 20*_stack50.tif_da_mask.tif -maxdepth 0 -exec gdal_edit {} -unsetnodata \;

find 20*_stack50.tif_da_mask.tif -maxdepth 0 -exec r_clean_range.py --infile={} --outfile={}_clean.tif --vmin=0 --vmax=100 \;
find 20*_stack50.tif_da_mask.tif_clean.tif -maxdepth 0 -exec gdal_edit {} -unsetnodata \;
```

## Generate the pair network

In a dedicated folder, typically in `working_dir/PAIRS/`, use AMSTer to generate a base network:

```shell
lns_all_Img.sh CSL_PATH PAIR_PATH S1
Prepa_MSBAS.sh PAIR_PATH Bperp Btemp PRIMARY_DATE
```

Create a Delaunay network, add yearly temporal baselines and optionally remove shorter temporal baselines:

```shell
DelaunayTable.sh -Ratio=r -BpMax=bp -BtMax=bt
amster_clean_table.py delaunay_table.txt --outfile=table_delaunay_clean.txt --it=0 --maxbp=maxbp --minbt=minbt
amster_generate_network.py table_multiyearly.txt allPairsListing.txt --maxbp=maxbp --maxbt=maxbt --minbt=minbt --restrain-nb
amster_merge_table.py table_multiyearly.txt table_delaunay_clean.txt --outfile=table_merge.txt
```

## ASP image correlation

Prepare the correlation parameters:

```shell
aspeo new aspsar
vi aspeo.toml
```

Launch the correlation:

```shell
aspeo pt aspeo.toml -v
```

## Post-processing

### Clean the results

```shell
stereo2export.py STEREO/ EXPORT/ --pairs=PAIRS/table_pairs.txt
```

## Time Series Inversion

Setup the directory:

```shell
export2nsbas.py EXPORT/ NSBAS/ --pairs=PAIRS/table_pairs.txt
```

Adapt the parameters:

```shell
vi NSBAS/V/input_inv_send
vi NSBAS/H/input_inv_send
```

Launch the correlation:

```shell
cd NSBAS/H
invers_pixel < input_inv_send
cd ../V
invers_pixel < input_inv_send
cd ../..
```

## Geocoding

Geocode the results using [`am_geocode`](amster#geocode-files), i.e in a GEOCODE directory.


