Polygonisation
==============

Polygonisation is a useful post-processing step on segmentation outputs.
This is a procedure whereby (thresholded) pixel-level segmentation rasters
are converted into sets of georeferenced polygons.

`GDAL <https://gdal.org/index.html>`_ provides convenient and fast parallelised 
scripts for performing raster thresholding and polygonisation, and this is the 
preferred method of generating shapefiles from segmentation rasters.

A typical workflow first performs thresholding on segmentation outputs::

    $ gdal_calc.py -A segmentation_output.tif --calc="A>0.5" --outfile=thresholded_segmentation.tif --NoDataValue=0

And finally produces polygons from segmented raster::

    $ gdal_polygonize.py thresholded_segmentation.tif -f "ESRI Shapefile" segmented_polygons.shp

Some complicating factors are:

Sometimes you may wish to polygonise rasters in JPEG2000 format. For this it is best to run GDAL with the 
JP2KAK driver enabled, as the stock drivers are extremely slow. This can be accomplished by using a docker 
container with GDAL built with these drivers.

A short working script for performing polygonisation is included in ``bin/polygonise.sh``. This relies on the 
``kak:v4`` docker image which is ``klokantech/gdal:2.3`` patched to include numpy (and thus enable ``gdal_calc.py``).

In a directory containing ``.tif`` segmentation outputs (mask rasters with probabilities [0, 1]) one can run 
for example::

    $ ./polygonise.sh -t 0.5 *.tif

To binarise the rasters using a threshold of 0.5 and create shapefile outputs for each one.

The script is repeated here for completeness::

    #!/bin/bash
    # get threshold value from -t flag
    while getopts t: flag
    do
        case "${flag}" in
            t) threshold=${OPTARG};;
        esac
        shift
        shift
    done
    echo "Threshold: $threshold"
    thresh_str="${threshold/./p}"
    for input_file in "$@"
    do	
        base_name=$(echo "$input_file" | cut -f 1 -d '.')
        # create output file names
        tif_out_tmp="${base_name}_thresh_${thresh_str}.tif"
        shp_out_file="${base_name}_thresh_${thresh_str}.shp"
        # set threshold query for gdal calc
        thresh_calc_str="A>=${threshold}"
        echo "${input_file} -> ${shp_out_file}"
        # perform theseholding and output to a temporary binary tif file
        docker run -ti -v $(pwd):/data kak:v4 gdal_calc.py -A $input_file --calc=$thresh_calc_str --outfile=$tif_out_tmp --NoDataValue=0
        # run gdal polygonize on the temporary tif to produce a shapefile
        docker run -ti -v $(pwd):/data kak:v4 gdal_polygonize.py $tif_out_tmp -f "ESRI Shapefile" $shp_out_file
        # remove the temporary tif file
        rm $tif_out_tmp
    done