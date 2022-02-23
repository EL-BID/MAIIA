""" interfaces/base.py
"""
import logging
import os
import shutil
import abc

import dask.array as da
import timbermafia as tm

import gim_cv.config as cfg

from pathlib import Path
from typing import Union, Any

from osgeo import gdal, osr, ogr

from gim_cv.preprocessing import rescale_image_array
from gim_cv.utils import hash_iterable
from gim_cv.exceptions import InvalidArrayError





log = logging.getLogger(__name__)


def cache_path_from_src_filename(src_filename, metadata=None) -> Path:
    """ return a cache directory for storing arrays based on img name """
    components = src_filename.parts[-1].split('.') # name, ext
    if metadata is not None:
        components.append(hash_iterable(metadata.items()))
    # get the name of the image dataset/shape file without extension
    _dp = '_'.join(components)
    return Path(f'{_dp}')


def rescale_metadata(metadata, new_w, new_h):
    """ derive a new metadata dict for when an raster is rescaled """
    metadata = metadata.copy()
    # scale image transform
    metadata['transform'] = metadata['transform'] * metadata['transform'].scale(
        (metadata['width'] / new_w),
        (metadata['height'] / new_h)
    )
    metadata['width'] = new_w
    metadata['height'] = new_h
    return metadata


class GeoArrayFileInterface(tm.Logged, metaclass=abc.ABCMeta):
    """ base interface class which anticipates an array linked to a serialised cache
        and some metadata which are derived from some source file (raster, shapefile...)
    """
    use_cache = False # default: don't cache arrays associated with rasters

    def __init__(self,
                 file_path,
                 metadata=None,
                 cache_dir=cfg.input_image_array_dir,
                 read_metadata=False,
                 ):
        self.metadata = metadata
        self.file_path = Path(file_path)
        self.cache_dir = cache_dir
        self.cache = ArrayCache(cache_path=self.default_cache_path())
        self.cache.metadata = self.metadata

    def default_cache_path(self):
        if self.cache_dir is None and self.use_cache:
            raise ValueError("use_cache is True, so cache_dir needs to be specified!")
        return Path(self.cache_dir) / cache_path_from_src_filename(self.file_path, self.metadata)

    @property
    def array(self):
        """ shortcut - read array from array cache class which provides
            access to array cache save/load methods
        """
        return self.cache.array

    @array.setter
    def array(self, arr):
        """ shortcut to cache class """
        self.cache.array = arr

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        if hasattr(self, 'cache'):
            self.cache.metadata = value

    @property
    def data_available(self):
        return True if self.array is not None and self.metadata is not None else False

    def validate_array(self, validation_fn=None):
        self.cache.validate(validation_fn)

    def load_array(self):
        """ delegates to read from raster or load from cache dep on use_cache attr """
        if self.use_cache:
            log.debug("Reading array from cache...")
            self.ensure_array_cached()
        else:
            log.debug("Reading array from source...")
            self.read_array_from_file()

    @property
    def data_available(self):
        return True if self.array is not None and self.metadata is not None else False

    def ensure_array_cached(self):
        self.log.debug(f"Ensuring zarr array is cached for file {self.file_path}...")
        try:
            self.cache.read()
            self.cache.validate()
        except Exception as e:
            self.log.debug(f"Loading from cache failed: {e}. Regenerating and saving array.")
            # read from source file
            self.read_array_from_file()
            # write to zarr formatted file
            self.cache.save(overwrite=True)
            # ensure arr attribute refers to zarr file
            self.cache.read()

    def rescale_array_and_metadata(self, rescale_factor, preserve_int=True):
        log.debug(f"Rescaling array by factor of {rescale_factor}...")
        self.array = rescale_image_array(array=self.array,
                                         sf=rescale_factor,
                                         preserve_int=preserve_int)
        self.metadata = rescale_metadata(self.metadata,
                                         new_w = self.array.shape[1],
                                         new_h = self.array.shape[0])


class BaseShapeInterface(GeoArrayFileInterface, tm.Logged):
    def __init__(self,
                 shp_path,
                 metadata=None,
                 cache_dir=None,
                 **kwargs
                 ):
        self.shp_path = Path(shp_path)
        self.metadata = metadata
        self.cache_dir = cache_dir
        self.cache = ArrayCache(cache_path=self.default_cache_path())

    @abc.abstractmethod
    def read_array_from_shapefile(self):
        pass

    # alias for compatibility with base class methods
    @property
    def read_array_from_file(self):
        return self.read_array_from_shapefile

    @property
    def shp_path(self):
        return self.file_path

    @shp_path.setter
    def shp_path(self, value):
        self.file_path = value

    def read_file(self):
        return self.read_array_from_file()


class BaseRasterInterface(GeoArrayFileInterface, tm.Logged):
    """ basic interface for classes reading files and extracting
        dask arrays, saving and loading these.
    """
    use_cache = False # default: don't cache arrays associated with rasters

    def __init__(self,
                 raster_path,
                 metadata=None,
                 cache_dir=cfg.input_image_array_dir,
                 read_metadata=False
                 ):
        self.raster_path = Path(raster_path)
        self.metadata = metadata
        self.cache_dir = cache_dir
        # if flagged, read metadata from existing file immediately
        if read_metadata:
            self.read_metadata_from_raster()
        self.cache = ArrayCache(cache_path=self.default_cache_path())

    @abc.abstractmethod
    def read_array_from_raster(self):
        pass

    # aliases for compatibility with base class methods
    @property
    def read_array_from_file(self):
        return self.read_array_from_raster

    @property
    def raster_path(self):
        return self.file_path

    @raster_path.setter
    def raster_path(self, value):
        self.file_path = value

    @abc.abstractmethod
    def read_metadata_from_raster(self):
        """ Metadata is extracted from the image file located at raster_path,
            and dumped to the metadata attribute here

            Necessary attributes are in georeferencedarraymixin

            Must set metadata attribute
        """
        pass

    @abc.abstractmethod
    def standardise_metadata(self, metadata):
        pass

    def read_raster(self):
        self.read_metadata_from_raster()
        return self.read_array_from_raster()

    @property
    def read_file(self):
        return self.read_raster

    @abc.abstractmethod
    def write_raster(self):
        pass

    def delete_raster(self):
        log.debug(f"Deleting raster file at {self.raster_path}...")
        os.remove(self.raster_path)

    def make_rescaled_copy(self, rescale_factor, file_path=None, preserve_int=True):
        """ create and return """
        rescaled_array = rescale_image_array(array=self.array,
                                             sf=rescale_factor,
                                             preserve_int=preserve_int)
        rescaled_metadata = rescale_metadata(self.metadata,
                                             new_w = rescaled_array.shape[1],
                                             new_h = rescaled_array.shape[0])
        if file_path is None:
            ext = str(self.raster_path).split('.')[-1]
            file_path = (
                self.file_path.parent /
                Path(self.raster_path.parts[-1].rstrip(f'.{ext}') + f'_rescale_{rescale_factor*100:.2g}pct.{ext}')
            )
        log.debug(f"Creating rescaled raster at {file_path}...")
        interface = type(self)(raster_path=file_path, metadata=rescaled_metadata, cache_dir=self.cache_dir)
        interface.array = rescaled_array
        return interface


class ArrayCache(tm.Logged):
    """ class to interface a dask array to a .zarr formatted cache
    """
    def __init__(self, array=None, cache_path=None):
        self.array = array
        self.cache_path = cache_path

    @property
    def is_cached(self):
        return self.cache_path.exists()

    def save(self, overwrite=False) -> None:
        if self.array is None:
            raise ValueError("Array not yet obtained!")
        if not self.cache_path.exists():
            log.debug(f"Cache path {self.cache_path} doesn't exist. Making directory...")
            self.cache_path.mkdir(parents=True)
        log.debug(f"Saving dask array to {self.cache_path}...")
        try:
            self.array.to_zarr(str(self.cache_path))
        except ValueError as v:
            if v.args[0] != "path '' contains an array" and not overwrite:
                raise
            if self.array is not None and overwrite:
                log.debug(f"overwriting array at {self.cache_path}")
                self.delete()
                self.array.to_zarr(str(self.cache_path))
            else:
                log.debug("Array already present. Skipping...")

    def read(self) -> Union[None, da.Array]:
        """ Attempts to load cached dask array for image, returning it
            if successful and None if not

            The array is expected to be located under self.cache_path
        """
        log.debug(f"Attempting to load dask array from {self.cache_path}...")
        try:
            self.array = da.from_zarr(str(self.cache_path))
            log.debug("Dask array loaded from cache.")
        except Exception as e:
            log.debug(f"Error loading array: {e}")
            raise
        return self.array

    def validate(self, validate_fn=None):
        """ check nonzero everywhere, 3 channels etc """
        #log.debug(f"validate fn is {validate_fn.__name__s}")
        if self.array.max() > 0.:
            pass
        if validate_fn is not None:
            #log.debug(f"applying validation check to input array with function: {validate_fn.__name__}")
            if validate_fn(self.array):
                return
            raise InvalidArrayError(f"array failed validation by function {validate_fn.__name__}!")

    def delete(self):
        """ clear cache """
        shutil.rmtree(self.cache_path)
