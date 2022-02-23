#import abc

#import building_age.config as cfg

#from pathlib import Path

#from osgeo import gdal, osr

#from building_age.interfaces.base import BaseArrayReader, GeoreferencedArrayMixin, InvalidArrayError

#with rasterio.open(raster_path) as dataset:
#    profile = dataset.profile
#new_w = rescaled_arr.shape[-2]
#new_h = rescaled_arr.shape[-1]



#def standardise_metadata(self, metadata):
#    """ https://rasterio.readthedocs.io/en/latest/topics/migrating-to-v1.html##
#
#        affine.Affine(a, b, c, == (c, a, b, f, d, e)
#                      d, e, f)
#    """
#    #gt = _metadata.pop('geo_transform')
#    #_metadata['transform'] = (gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]) 
#    #self.metadata
#    #log.debug("TODO - write standardise_metadata!")
#    return metadata

  
#class GDALImageReader(BaseImageReader, tm.Logged):
#    """ Implements getting metadata from image with gdal library
#    """
#    
#    def open_image_dataset(self):
#        """ open a dataset with GDAL library, returning a gdal.Dataset object
#        """
#        self._ds = gdal.Open(str(self.src_path))
#        assert self._ds is not None, "Can't open image file to retrieve metadata!"      
#        return self._ds
#    
#    def close_image_dataset(self):
#        self._ds = None
#    
#    @property
#    def ds(self):
#        return self._ds
#    
#    @property
#    def vma(self):
#        return self.ds.GetVirtualMemArray().transpose(1,2,0)    
#    
#    def get_metadata(self):
#        """ Populate the metadata dict by reading the file with GDAL
#        """
#        ds = self.open_image_dataset()
#        _metadata = {'geo_transform' : ds.GetGeoTransform(),
#                     'projection': ds.GetProjection(),
#                     'raster_x_size' : ds.RasterXSize,
#                     'raster_y_size' : ds.RasterYSize}
#        ds = None
#        self.metadata = self.standardise_metadata(_metadata)
#        
#    def standardise_metadata(self, metadata):
#        """ https://rasterio.readthedocs.io/en/latest/topics/migrating-to-v1.html
#
#            affine.Affine(a, b, c, == (c, a, b, f, d, e)
#                          d, e, f)
#        """
#        #gt = _metadata.pop('geo_transform')
#        #_metadata['transform'] = (gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]) 
#        #self.metadata
#        log.debug("TODO - write standardise_metadata!")
#        return metadata

    
   
#def get_metadata(self):
#    """ Populate the metadata dict by reading the file with GDAL
#    """
#    with rasterio.open(self.src_path, 'r') as src:
#        self.metadata = src.profile               
#    self.metadata = self.standardise_metadata(self.metadata)

#def get_metadata(self):
#    """ Populate the metadata dict by reading the file with GDAL
#    """
#    ds = self.open_image_dataset()
#    _metadata = {'geo_transform' : ds.GetGeoTransform(),
#                 'projection': ds.GetProjection(),
#                 'raster_x_size' : ds.RasterXSize,
#                 'raster_y_size' : ds.RasterYSize}
#    ds = None
#    self.metadata = self.standardise_metadata(_metadata)      
#    
#def standardise_metadata(self, metadata):
#    """ 
#    """
#    log.debug("TODO - write standardise metadata!")
#   return metadata    