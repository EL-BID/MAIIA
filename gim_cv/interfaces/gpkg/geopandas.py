import sys
import hashlib

import timbermafia as tm
import numpy as np
import dask.array as da

import gim_cv.config as cfg

from typing import Union
from pathlib import Path

from osgeo import gdal, osr, ogr

from gim_cv.interfaces.base import BaseShapeInterface

import logging

log = logging.getLogger(__name__)


class ShapeReader(BaseShapeInterface, tm.Logged):
    """ Base class for reading a shapefile located at some src_path.

        Open/closing with GDAL exposes features in the shape_data attribute
    """
    shapefile_driver_name = "ESRI Shapefile"

    def open_shapefile(self, writable=0):
        """ open a shapefile with OGR, setting private attribute to access
            ogr.Datasource object
        """
        writable=int(writable)
        self.driver = ogr.GetDriverByName(self.shapefile_driver_name)
        self._shp = self.driver.Open(
            str(self.shp_path), writable
        )
        assert self._shp is not None, "Can't open shapefile!"
        return self._shp

    def close_shapefile(self):
        self._shp = None

    @property
    def shape_data(self):
        return self._shp

    @property
    def layer(self):
        """ Shortcut property to get layer of features from shapefile """
        return self.shape_data.GetLayer()

    @property
    def geo_uid_tuple(self):
        """ generate a positive definite unique id for cached array file name

            hash tuple instead of XOR to prevent equal attr values on e.g. raster x/y size cancelling

            https://stackoverflow.com/questions/18766535/positive-integer-from-python-hash-function
        """
        # need to deal with numerical precision errors on floats; round to 2 DP
        gt = '_'.join(f"{nr:.2f}" for nr in self.geo_transform)
        return (self.spatial_reference.GetName(), gt, str(self.raster_x_size), str(self.raster_y_size))

    @property
    def feature_count(self):
        """ Shortcut property to get feature count from layer in shapefile """
        return self.layer.GetFeatureCount()

    def add_FID_column(overwrite:bool=True) -> None:
        """ Write the FID value explicitly to a shapefile database
            (rather than inferring this automatically) as a new column

            This should be an integer ascending from 1 depending on
            how many records are present in the shapefile

            Signature:
                overwrite:   bool, overwrite FIDs or not
        """
        self.open_shapefile(writable=1)
        # add an ID field if necessary
        fldDef = ogr.FieldDefn('FID', ogr.OFTInteger64)
        # if it's not there
        if self.layer.GetLayerDefn().GetFieldIndex("FID") < 0:
            self.log.debug("creating FID column")
            self.layer.CreateField(fldDef)
        # skip creation if present, also return now if no overwrite
        else:
            self.log.debug("skipping FID column creation...")
            if not overwrite:
                return
        # write FID values to shp file
        for Ft in layer:
            ThisID = int(Ft.GetFID())
            Ft.SetField('FID', ThisID)          # Write the FID to the ID field
            self.layer.SetFeature(Ft)               # update the feature
        self.close_shapefile()


class BinaryMaskShapeReader(ShapeReader):
    """ concrete implementation of MaskReader that associates binary masks with shapefiles
        provided a given geo transform/projection/extent
    """
    def read_array_from_shapefile(self):
        """ read with GDAL and burn features onto binary raster mask """
        if self.metadata is None:
            raise AttributeError("metadata attribute not supplied to shape reader!")
        try:
            self.log.debug(f"Reading array from file {self.shp_path}...")
            self.open_shapefile()
            prj = self.metadata['crs'].wkt
            gt = self.metadata['transform'].to_gdal()
            sx = self.metadata['width']
            sy = self.metadata['height']
            self.array = rasterise_shapefile(layer=self.layer,
                                           projection=prj,
                                           geo_transform=gt,
                                           raster_x_size=sx,
                                           raster_y_size=sy)
            return self.array
        except:
            self.log.error("Failed to rasterise shapefile!")
            raise
        finally:
            self.close_shapefile()


class LabelledMaskShapeReader(ShapeReader):
    """ concrete implementation of MaskReader that associates masks with shapefiles
        provided a given geo transform/projection/extent.

        masks are integer valued, where integer identifiers are taken directly from
        an attribute of the shapefile. these can correspond to e.g. a unique
        identifier for each object
    """

    def __init__(self,
                 label_attr='fid',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_attr = label_attr

    def read_array_from_shapefile(self):
        """ read with GDAL and burn features onto an integer-valued raster mask """
        try:
            self.log.debug(f"Reading array from file {self.shppath}...")
            self.open_shapefile()
            prj = self.metadata['crs'].wkt
            gt = self.metadata['transform'].to_gdal()
            sx = self.metadata['width']
            sy = self.metadata['height']
            self.array = rasterise_shapefile(layer=self.layer,
                                             projection=prj,
                                             geo_transform=gt,
                                             raster_x_size=sx,
                                             raster_y_size=sy,
                                             dtype=gdal.GDT_UInt32,
                                             options=[f'ATTRIBUTE={self.label_attr}'])
            return self.array
        except:
            self.log.error("Failed to rasterise shapefile!")
            raise
        finally:
            self.close_shapefile()


def rasterise_shapefile(layer,
                        projection,
                        geo_transform,
                        raster_x_size,
                        raster_y_size,
                        dtype=gdal.GDT_Byte,
                        nodata_val=0,
                        options = ["ALL_TOUCHED=TRUE"],
                        as_dask=True) -> Union[np.array, da.Array]:
    """
    Rasterise one layer of a shapefile to a binary mask array with GDAL

    Creates a pixel grid with array with binary 1/0 values (by default)
    corresponding to the presence/lack of objects.

    Returns either numpy or dask array depending on as_dask parameter.

    for integer labels in mask call with:
    dtype=gdal.GDT_UInt32, options=[..., 'ATTRIBUTE=fid']

    Parameters
    ----------
    dtype:
        the gdal dtype for values burned onto the raster
    nodata_val:
        the value to appear where there is no shape data

    Returns
    -------
    :obj:`numpy.ndarray` or :obj:`dask.array.Array`:
        Mask array with dimensions raster_x_size, raster_y_size

    based on:
    github.com/terrai/rastercube/blob/master/rastercube/datasources/
    shputils.py
    """
    log.debug("Creating mask array...")
    # create a new raster in memory
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create(
        '',
        raster_x_size,
        raster_y_size,
        1,
        dtype
    )
    # set the geo transform to the same as the input image
    mem_raster.SetProjection(projection)
    # set the geo transform to the same as the input image
    mem_raster.SetGeoTransform(geo_transform)
    # get the (single) raster band in the new dataset
    mem_band = mem_raster.GetRasterBand(1)
    # write zeros everywhere where no data
    mem_band.Fill(nodata_val)
    mem_band.SetNoDataValue(nodata_val)
    # burn values
    # http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
    # http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer
    # -doesnt-burn-all-polygons-to-raster
    # ---- this is slow as hell
    #for bid in self.building_ids:
    #    if int(bid) % 10 == 0:
    #        print(bid)
    #    err = gdal.RasterizeLayer(mem_raster, [1], self.layer, burn_values=[bid], options=options)
    #    assert err == gdal.CE_None, str(err)
    err = gdal.RasterizeLayer(
        mem_raster,
        [1], # band list
        layer, # source layer
        None, # transformer
        None, # transformer_arg
        [1], # burn_values
        options=options
    )
    assert err == gdal.CE_None, str(err)
    if as_dask:
        return da.from_array(mem_raster.ReadAsArray())
    return mem_raster.ReadAsArray()
