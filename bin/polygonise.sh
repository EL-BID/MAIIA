#!/bin/bash
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
	tif_out_tmp="${base_name}_thresh_${thresh_str}.tif"
	shp_out_file="${base_name}_thresh_${thresh_str}.shp"
	thresh_calc_str="A>=${threshold}"
	echo "${input_file} -> ${shp_out_file}"
	docker run -ti -v $(pwd):/data kak:v4 gdal_calc.py -A $input_file --calc=$thresh_calc_str --outfile=$tif_out_tmp --NoDataValue=0
	docker run -ti -v $(pwd):/data kak:v4 gdal_polygonize.py $tif_out_tmp -f "ESRI Shapefile" $shp_out_file
	rm $tif_out_tmp
done
