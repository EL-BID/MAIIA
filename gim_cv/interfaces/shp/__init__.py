import importlib
import timbermafia as tm
import gim_cv.config as cfg

from pathlib import Path

import logging

log = logging.getLogger(__name__)


def add_id_column_to_shp_file(shp_file:str, overwrite:bool=True) -> None:
    """ Write the FID value explicitly to a shapefile database
        (rather than inferring this automatically) as a new column

        Signature:
            shp_file:    a shape file to be read by ogr
            overwrite:   bool, overwrite FIDs or not
    """
    assert Path(shp_file).exists()
    daShapefile = str(shp_file) # input shapefile, field ID must already exist
    driver      = ogr.GetDriverByName('ESRI Shapefile')
    dataSource  = driver.Open(daShapefile, 1) # 1 for writable
    layer       = dataSource.GetLayer()       # get the layer for this datasource

    # add an ID field if necessary
    fldDef = ogr.FieldDefn('FID', ogr.OFTInteger64)
    # if it's not there
    if layer.GetLayerDefn().GetFieldIndex("FID") < 0:
        log.debug("creating FID column")
        layer.CreateField(fldDef)
    # skip creation if present, also return now if no overwrite
    else:
        log.debug("skipping FID column creation...")
        if not overwrite:
            return
    # write FID values to shp file
    for Ft in layer:
        ThisID = int(Ft.GetFID())
        Ft.SetField('FID',ThisID)          # Write the FID to the ID field
        layer.SetFeature(Ft)               # update the feature
    dataSource = None


# -- readers

# look up the preferred default reader class using the config. currently, this should be named JP2Reader and implemented
# in a directory with the name of the jp2_reader variable.
# for example, set jp2_reader = rasterio and implement JP2ImageReader in rasterio.py in this directory
BinaryMaskShapeReader = getattr(importlib.import_module(f'gim_cv.interfaces.shp.{cfg.shp_reader}'), 'BinaryMaskShapeReader')
LabelledMaskShapeReader = getattr(importlib.import_module(f'gim_cv.interfaces.shp.{cfg.shp_reader}'), 'LabelledMaskShapeReader')

class BinaryMaskReader(BinaryMaskShapeReader, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = cfg.input_binary_mask_array_dir
        # set array caching policy
        self.use_cache = cfg.cache_shp and cfg.cache_input_binary_mask_arrays
        super().__init__(*args, **kwargs)


class LabelledMaskReader(LabelledMaskShapeReader, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = cfg.input_labelled_mask_array_dir
        # set array caching policy
        self.use_cache = cfg.cache_shp and cfg.cache_input_labelled_mask_arrays
        super().__init__(*args, **kwargs)
