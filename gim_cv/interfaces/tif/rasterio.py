import gc
import warnings

import rasterio
import numpy as np
import dask.array as da
import timbermafia as tm

import gim_cv.config as cfg

from threading import Lock

from cached_property import cached_property
from dask.base import tokenize
from dask import is_dask_collection
from rasterio.transform import Affine, guard_transform
from rasterio.windows import Window
#from xarray import DataArray

from gim_cv.utils import require_attr_true
from gim_cv.interfaces.base import BaseRasterInterface

import logging

log = logging.getLogger(__name__)


class RasterInterface(BaseRasterInterface, tm.Logged):
    use_cache = False
    """ Implements parallel read of tif with RasterIO """

    def standardise_metadata(self, metadata):
        """ rasterio is the base standard for metadata """
        return metadata

    def read_metadata_from_raster(self):
        with rasterio.open(self.raster_path, 'r') as src:
            profile = src.profile
        self.metadata = self.standardise_metadata(profile)
        return self.metadata

    def read_array_from_raster(self, *args, **kwargs):
        """ add rechunk as blocks are typically quite small (O(1MB))
        """
        log.debug(f"Reading raster from file {self.raster_path}...")
        self.array = read_raster(self.raster_path).transpose(1, 2, 0) # swap to channels last
        log.debug(f"Done reading raster from file {self.raster_path}")
        return self.array

    def write_raster(self, overwrite=False, *args, **kwargs):
        assert self.data_available, ("Can't write raster - array and metadata not set!")
        log.debug(f"Writing raster to file {self.raster_path}...")
        if self.raster_path.exists():
            if not overwrite:
                log.debug(f"File {self.raster_path} already exists! Skipping...")
                return
        else:
            self.raster_path.parent.mkdir(parents=True, exist_ok=True)
        profile = self.metadata.copy()
        profile.update(dtype=self.array.dtype,
                       count=self.array.shape[2], # bands
                       compress='lzw')
        write_raster(
            self.raster_path,
            self.array.transpose(2, 0, 1), # swap to channels first
            **profile
        )
        log.debug(f"Done writing raster at {self.raster_path}!")

## workhorse functions
## mostly adapted from dask-rasterio
def read_raster_band(path,
                     band=1,
                     target_chunk_size=cfg.da_chunk_size,
                     natural_chunk_multiplier=None):
    """Read a raster band and return a Dask array
    Arguments:
        path {string} -- path to the raster file
    Keyword Arguments:
        band {int} -- number of band to read (default: {1})
        block_size {int} -- block size multiplier (default: {1})
        natural_chunk_multiplier -- if target_chunk_size is None,
                                    fall back on this x natural chunks
                                    (along each axis, so mult^2 natural chunks)
                                    if this is None, just read natural blocks
    """
    log.debug(f"read raster band {band}")
    def read_window(raster_path, window, band):
        with rasterio.open(raster_path) as src:
            return src.read(band, window=window)

    def resize_window(window, block_size):
        return Window(
            col_off=window.col_off * block_size,
            row_off=window.row_off * block_size,
            width=window.width * block_size,
            height=window.height * block_size)

    def block_windows(dataset, band, block_size):
        return [(pos, resize_window(win, block_size))
                for pos, win in dataset.block_windows(band)]

    with rasterio.open(path) as src:
        dtype = src.dtypes[band - 1]
        shape = src.shape
        nat_blk_h, nat_blk_w = src.block_shapes[band - 1]
        width, height = src.meta['width'], src.meta['height']
        log.debug(f"dask reading raster with width, height: {width}, {height}")
        log.debug(f"natural block shape is width, height: {nat_blk_w}, {nat_blk_h}")
        # if no chunk size specified, read natural bllocks, optionally * some multiplier
        # (same in both dimensions)
        if target_chunk_size is None:
            mult = 1 if natural_chunk_multiplier is None else natural_chunk_multiplier
            n_cols_chunk, n_rows_chunk = mult*nat_blk_w, mult*nat_blk_h
        # otherwise, adaptively assemble natural blocks to approximately
        # form dask chunks of w, h : w*h ~= target_chunk_size^2
        else:
            # if the largest dimension of the image is less than the
            # target chunk size, just read the whole band as one block
            if max(width, height) <= target_chunk_size:
                target_chunk_size = max(width, height) # generalise this to x != y
            # try to get as many natural blocks
            # natural block size is h*w
            total_size = target_chunk_size * target_chunk_size
            log.debug(f"total size is: {total_size}")
            if nat_blk_w > nat_blk_h:
                n_cols_chunk = nat_blk_w
                # take however many vertical blocks so that
                # the block size is closest to total size
                n_rows_chunk = min(int(total_size / nat_blk_w), height)
                log.debug(f"n_rows_chunk calculated as min({total_size}/{nat_blk_w}, {height}): {n_rows_chunk}")
            else:
                # ditto for horizontal blocks
                n_cols_chunk = min(int(total_size / nat_blk_h), width)
                log.debug(f"n_cols_chunk calculated as min({total_size}/{nat_blk_h}, {width}): {n_cols_chunk}")
                n_rows_chunk = nat_blk_h
        chunk_shape = (n_rows_chunk, n_cols_chunk)
        log.debug(f"reading raster with chunk shape: {chunk_shape}")
        vchunks = range(0, shape[0], chunk_shape[0])
        hchunks = range(0, shape[1], chunk_shape[1])
        name = 'raster-{}'.format(tokenize(path, band, chunk_shape))
        #blocks = block_windows(src, band, block_size)
        chunk_shape = da.core.normalize_chunks(chunk_shape, shape=src.shape)
    # build the dask task graph manually
    dsk = {}
    for i, vcs in enumerate(vchunks):
        for j, hcs in enumerate(hchunks):
            window = Window(hcs, vcs,
                            min(n_cols_chunk,  shape[1] - hcs),
                            min(n_rows_chunk,  shape[0] - vcs))
            dsk[(name, i, j)] = (read_window, path, window, band)
    # assemble tasks into array
    _darr = da.Array(dsk, name, chunk_shape, dtype, shape=shape)
    return _darr


def get_band_count(raster_path):
    """Read raster band count"""
    with rasterio.open(raster_path) as src:
        return src.count


def read_raster(path,
                band=None,
                target_chunk_size=cfg.da_chunk_size,
                natural_chunk_multiplier=None):
    """Read all or some bands from raster
    Arguments:
        path {string} -- path to raster file
    Keyword Arguments:
        band {int, iterable(int)} -- band number or iterable of bands.
            When passing None, it reads all bands (default: {None})
        block_size {int} -- block size multiplier (default: {1})
    Returns:
        dask.array.Array -- a Dask array
    """

    if isinstance(band, int):
        return read_raster_band(path,
                                band=band,
                                target_chunk_size=target_chunk_size,
                                natural_chunk_multiplier=natural_chunk_multiplier)
    else:
        if band is None:
            bands = range(1, get_band_count(path) + 1)
        else:
            bands = list(band)
        return da.stack([
            read_raster_band(path,
                             band=band,
                             target_chunk_size=target_chunk_size,
                             natural_chunk_multiplier=natural_chunk_multiplier)
            for band in bands
        ])



class RasterioDataset(tm.Logged):
    """Rasterio wrapper to allow dask.array.store to do window saving.
    Example:
        >> rows = cols = 21696
        >> a = da.ones((4, rows, cols), dtype=np.float64, chunks=(1, 4096, 4096) )
        >> a = a * np.array([255., 255., 255., 255.])[:, None, None]
        >> a = a.astype(np.uint8)
        >> with RasterioDataset('test.tif', 'w', driver='GTiff', width=cols, height=rows, count=4, dtype=np.uint8) as r_file:
        ..    da.store(a, r_file, lock=True)
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.dataset = None

    def __setitem__(self, key, item):
        """Put the data chunk in the image"""
        if len(key) == 3:
            index_range, y, x = key
            indexes = list(
                range(index_range.start + 1, index_range.stop + 1,
                      index_range.step or 1))
        else:
            indexes = 1
            y, x = key

        chy_off = y.start
        chy = y.stop - y.start
        chx_off = x.start
        chx = x.stop - x.start

        self.dataset.write(
            item, window=Window(chx_off, chy_off, chx, chy), indexes=indexes)
        # ensure memory cleared
        del item, indexes
        gc.collect()

    def __enter__(self):
        """Enter method"""
        self.dataset = rasterio.open(*self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method"""
        self.dataset.close()


def write_raster(path, array, **kwargs):
    """Write a dask array to a raster file
    If array is 2d, write array on band 1.
    If array is 3d, write data on each band
    Arguments:
        path {string} -- path of raster to write
        array {dask.array.Array} -- band array
        kwargs {dict} -- keyword arguments to delegate to rasterio.open
    Examples:
        # Write a single band raster
        >> red_band = read_raster_band("test.tif", band=1)
        >> write_raster("new.tif", red_band)
        # Write a multiband raster
        >> img = read_raster("test.tif")
        >> new_img = process(img)
        >> write_raster("new.tif", new_img)
    """
    if len(array.shape) != 2 and len(array.shape) != 3:
        raise TypeError('invalid shape (must be either 2d or 3d)')

    if is_dask_collection(array):
        with RasterioDataset(path, 'w', **kwargs) as dst:
            da.store(array, dst, lock=True)
    else:
        with rasterio.open(path, 'w', **kwargs) as dst:
            if len(array.shape) == 2:
                dst.write(array, 1)
            else:
                dst.write(array)


def pad(array, transform, pad_width, mode=None, **kwargs):
    """ Drop-in replacement patch for rasterio.pad which works also on dask arrays
    """
    if isinstance(array, np.ndarray):
        lib = np
    elif isinstance(array, da.Array):
        lib = da
    else:
        raise TypeError(f"Type {type(array)} of {array} not understood in pad."
                        " Should be a numpy/dask array.")
    print("trans", transform)
    transform = guard_transform(transform)
    padded_array = lib.pad(array, pad_width, mode, **kwargs)
    # transform is (x_res, 0, x_0, 0, -y_res, y_0)
    padded_trans = list(transform)
    pw_arr = np.squeeze(pad_width)
    # int: all sides in all dimensions padded by pad_width
    if isinstance(pad_width, int):
        pad_width_left_x = pad_width
        pad_width_top_y = pad_width
    # same syntax for all dimensions e.g.:
    # (0,1) => 0 padding @ beginning (top, left), 1 padding @ end (right, bottom) - for _all_ dims
    elif len(np.squeeze(pw_arr).shape) == 1:
        pad_width_left_x = pw_arr[0]
        pad_width_top_y = pw_arr[0]
    # other scenario: 3 dimensions specify padding separately for rows, cols, channels
    elif pw_arr.shape[0] == 3:
            # x -> cols, y -> rows
            pad_width_left_x = pw_arr[1, 0]
            pad_width_top_y =  pw_arr[0, 0]
    else:
        raise ValueError(f"padding width specification shape {pw_arr.shape} not understood")
    padded_trans[2] -= pad_width_left_x * padded_trans[0]
    padded_trans[5] -= pad_width_top_y * padded_trans[4]
    padded_trans = Affine(*padded_trans[:6])
    print("padded", padded_trans)
    return padded_array, padded_trans

def pad_raster_to_window(array, transform, min_rows, min_cols, constant_values=1):
    """Pad an array and its corresponding transform to match at least
       target_rows, target_cols along its first two dimensions
    """
    padding_rows = max(min_rows - array.shape[0], 0)
    padding_cols = max(min_cols - array.shape[1], 0)
    return pad(
        array, transform,
        pad_width=(
           (0, padding_rows), # start, end (both axes)
           (0, padding_cols),
           (0, 0)
        ),
        mode='constant',
        constant_values=constant_values
    )
