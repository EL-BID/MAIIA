
import os


# Define the x/y max, min and the center coordinates
xmin = '122000'
xmax = '138000'
ymin = '173000'
ymax = '188000'
xc = '130000'
yc = '180500'
# nomenclature is xmin ymin xmax ymax
# area numbering
# ---------
# | 1 | 2 |
# |---|---|
# | 3 | 4 |
# ---------
inCoords = {1:[xmin, yc, xc, ymax], 
            2:[xc, yc, xmax, ymax],
            3:[xmin, ymin, xc, yc],
            4:[xc, ymin, xmax, yc]}

input_layer = r'C:\Users\lnlembandognje\Downloads\1979-90\combined_1979.vrt'
output_folder = r'C:\Users\lnlembandognje\Downloads\1979-90\output'

cmdLines = ''
for coord in inCoords:
    cmdLines += 'gdalwarp -te %s %s %s\n'%(" ".join(inCoords[coord]), input_layer, os.path.join(output_folder, 'Aalst_Area_%s.tif'%coord))

with open(os.path.join(output_folder,'create_files.bat'), 'w') as fout:
	fout.write(cmdLines)