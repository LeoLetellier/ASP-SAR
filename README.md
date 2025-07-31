# ASP-SAR

Processing tool package written in bash and python for image correlation of coregistered SAR images using the AMES stereo toolbox (https://stereopipeline.readthedocs.io/en/latest/). The coregistration is done using the AMSTer toolbox(https://github.com/AMSTerUsers/AMSTer_Distribution/).\
The processing chain additionally compute the time series analysis with NSBAS .

![PNTS logo](logo-pnts.jpg)

# Workflow

1. AMSTer Coregistration

```bash
ALL2GIF.sh YYYYMMDD /full-path-to-LaunchParam.txt 100 100
```

Launch command in AMSTer/SAR_CSL folder, coregistrated images stored in AMSTer/SAR_SM/AMPLITUDE folder.

2. Initialize ASP-SAR

```bash
amster2aspsar.py /full-path/AMSTer/SAR_SM/AMPLITUDES/SAT/TRK/REGION /ASP-SAR/working-dir [--s1]
```

3. Generate pair network

For example, in /ASP-SAR/working-dir/PAIRS

```bash
lns_All_Img.sh CSL-dir pair-dir
```

```bash
Prepa_MSBAS.sh pair-dir Bpmax Btmax
```

Eventually use Delaunay triangulation:

```bash
DelaunayTable.sh -Ratio=r -BpMax=bp -BtMax=bt
```

You can copy your table of choice as table_pairs.txt.

Display the network:
````bash
amster_plot_network.py table.txt
````

Add a new pair to the table:
````bash
amster_table_add.py table.txt d1-d2 d2-d3 ...
````

Remove a pair from the table:

* Open the table in VI
* Cursor on the line to delete and `dd`


4. Image Correlation using ASP (through aspeo)

Generate the parameter file:

```bash
aspeo new preset
```

Update the parameters of interest

Launch the stereo process:

```bash
aspeo pt aspeo.toml -v
```

5. Export the results for NSBAS

```bash
aspsar2nsbas.py asp-sar-dir
```

6. Run TS inversion in WORKING_DIR/NSBAS_PROCESS/MASKED/H|V

Update the parameters of inversion: `input_inv_send`

Launch the inversion:

```bash
invers_pixel < input_inv_send
```

7. Geocode results

You can create a GEOCODE directory to store geocoded results.

Choose one of the date directory to host the geocoding steps and check the geocoding parameters in LaunchMTparams.txt.

Note that the sm run must have corrected filepaths. Run `RenamePathAfterMove_in_SAR_SM_AMPLITUDES.sh SAT` if not the case.

```bash
am_geocode.py file --amster=sm/amplitude/trk/region/one-of-the-dates/ --outdir=../../GEOCODE/[HV]
```

If conflicts of python version, you can run once `am_geocode`, then in the sm/amplitude/date dir run `ReGeocode_AmpliSeries.sh /full-path-to-params`, then run `am_geocode` again to get back the results.

GDAL mutli-band rasters will be split by band. To reconstruct the cube, check `merge_star.py`.
