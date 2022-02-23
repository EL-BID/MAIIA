import importlib
import logging

import timbermafia as tm

import gim_cv.config as cfg

from pathlib import Path


log = logging.getLogger(__name__)

# -- readers

# look up the preferred default reader class using the config. currently, this should be named JP2Reader and implemented
# in a directory with the name of the jp2_reader variable.
# for example, set jp2_reader = rasterio and implement JP2ImageReader in rasterio.py in this directory
RasterInterface = getattr(importlib.import_module(f'gim_cv.interfaces.tif.{cfg.tif_reader}'), 'RasterInterface')

# set defaults for rasterinterface subclasses - just clones with different cache directories
# and cache on/off switches depending on image/mask and read/write
class ImageReader(RasterInterface, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = cfg.input_image_array_dir
        # set array caching policy
        self.use_cache = cfg.cache_tif and cfg.cache_input_image_arrays
        super().__init__(*args, **kwargs)


# tif also used for masks - change default save dir, and these have their own metadata so dont req it explicitly
class BinaryMaskReader(RasterInterface, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = cfg.input_binary_mask_array_dir
        # set array caching policy
        self.use_cache = cfg.cache_tif and cfg.cache_input_binary_mask_arrays
        super().__init__(*args, **kwargs)


class LabelledMaskReader(RasterInterface, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = cfg.input_labelled_mask_array_dir
        # set array caching policy
        self.use_cache = cfg.cache_tif and cfg.cache_input_labelled_mask_arrays
        super().__init__(*args, **kwargs)

# -- writers

class ImageWriter(RasterInterface, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = cfg.output_image_array_dir
        # set array caching policy
        self.use_cache = cfg.cache_tif and cfg.cache_output_image_arrays
        super().__init__(*args, **kwargs)


class BinaryMaskWriter(RasterInterface, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' not in kwargs:
            kwargs['cache_dir'] = cfg.output_binary_mask_array_dir
        self.use_cache = cfg.cache_tif and cfg.cache_output_binary_mask_arrays
        super().__init__(*args, **kwargs)


class LabelledMaskWriter(RasterInterface, tm.Logged):
    def __init__(self, *args, **kwargs):
        if 'cache_dir' in kwargs:
            kwargs['cache_dir'] = cfg.output_labelled_mask_array_dir
        self.use_cache = cfg.cache_tif and cfg.cache_output_labelled_mask_arrays
        super().__init__(*args, **kwargs)
