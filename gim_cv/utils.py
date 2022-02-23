""" utils.py

    Miscellaneous utility functions used elsewhere
"""
import hashlib
import threading
import multiprocessing
import logging

import numpy as np
import matplotlib.pyplot as plt
import dask
import os
import sys
import tensorflow as tf
import dask.array as da
import timbermafia as tm

import gim_cv.config as cfg

from functools import wraps
from pathlib import Path
from time import perf_counter as pc
from typing import Union, Any

from osgeo import gdal, osr, ogr
from scipy.sparse import coo_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable



log = logging.getLogger(__name__)


class RegisteredAttribute(tm.Logged):
    """ descriptor """
    def __init__(self, name, registry):
        self.name = name
        self.registry = registry
    def __set__(self, instance, value):
        #log.debug(f"RegisteredAttribute {instance} {value}")
        # if the instance is already tracked, dont modify registry
        #if instance in self.registry.values():
        #    log.debug(f"skip {instance}")
        #    return
        # otherwise if the attribute value is already tracking an instance, raise error
        #elif value in self.registry.keys():
        #    #log.debug(self.registry, value)
        #    raise AttributeError(f"{value} already registered")
        # otherwise update the registry
        #else:
        instance.__dict__[self.name] = value
        self.registry[value] = instance

    #def __get__(self, instance, owner):
    #    if instance is None:
    #        return self
    #    else:
    #        return getattr(instance, self.name)

    
def parse_kwarg_str(kwarg_str):
    """
    Parses string from stdin into dict of kwargs
    
    Parameters
    ----------
    kwarg_str: str
        String with comma-delineated kwarg pairs, like 'x=1,y=2.5,z=chips'.
        
    Returns
    -------
    dict:
        {'x':1, 'y':2.5, 'z':'chips'} with numeric quantities type-guessed into int/floats
    """
    kwargs = {}
    for kwarg_str in kwarg_str.split(','):
        k, _v = kwarg_str.split('=')
        if '.' in _v:
            v = float(_v)
        else:
            try:
                v = int(_v)
            except ValueError:
                v = str(_v)
        kwargs[k] = v
    return kwargs
    
    
def require_attr_true(attr):
    """
    Decorator to enforce an attribute is true when calling a method

    Parameters
    ----------
    attr: 
        String name of attribute which must be True
    """
    assert isinstance(attr, str), "only single string attr names supported"
    def f_if_attr_true(f):
        @wraps(f)
        def inner(self, *args, **kwargs):
            if getattr(self, attr):
                return f(self, *args, **kwargs)
            raise AttributeError(f"Attribute {attr} not True!")
        return inner
    return f_if_attr_true

def hash_iterable(iterable):
    """ 
    Generate a positive definite unique id for cached array file name

    Hash tuple instead of XOR to prevent equal attr values on e.g. raster x/y size cancelling

    Parameters
    ----------
    iterable:
        Iterable of values to be hashed

    Returns
    -------
    Hex string of hashed values

    References
    ----------
    https://stackoverflow.com/questions/18766535/positive-integer-from-python-hash-function
    """
    m = hashlib.md5()
    for s in iterable:
        m.update(str(s).encode('utf-8'))
    return m.hexdigest()

#def tuple_hash_pos_def(tup):
#    """ positive definite unique id """
#    return hash(tuple(tup) % ((sys.maxsize + 1) * 2))

def free_from_white_pixels(array):
    """
    Returns True if an array contains no RGB values (255, 255, 255)

    Useful if (255, 255, 255) indicates missing data

    Parameters
    ----------
    array:
        An image array with the last axis being the channel dimension

    Returns
    -------
    True if no white pixels, False otherwise.
    """
    sum_ = array.sum(axis=-1).max().compute()
    return True if sum_ < 765 else False


def bounding_box_from_ixs(img, ixs):
    """ 
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Parameters
    ----------
    img: 2D numpy array

    Returns
    -------
    Tuple of slices for (row indices), (column indices) of box where arr == value
    """
    return [slice(min(ixarr), max(ixarr) + 1) for ixarr in ixs]


def bounding_box_equals(img, value=0):
    """ 
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Parameters
    ----------
    img: 2D numpy array

    Returns
    -------
    Tuple of slices for (row indices), (column indices) of box where arr == value
    """
    ixs = np.where(img == value)
    return [slice(min(ixarr), max(ixarr) + 1) for ixarr in ixs]


def bounding_box_nonzero(img):
    """ 
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Parameters
    ----------
    img:
        2D numpy array

    Returns
    -------
    A tuple of slices for (row indices), (column indices) of box where arr != 1
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return slice(rmin, rmax+1), slice(cmin, cmax+1)


def onehot_sparse(a):
    """ 
    Fast solution for label->one-hot encoding SPARSE - see:
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy

    Parameters
    ----------
    a : 
        A numpy array of values representing label data

    Returns
    -------
    One-hot encoded array
    """
    N = a.size
    L = a.max()+1
    data = np.ones(N,dtype=int)
    return coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,L))


def onehot_initialization_v2(a):
    """ 
    Fast solution for label->one-hot encoding, see:
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy

    Parameters
    ----------
    a : 
        A numpy array of values representing label data

    Returns
    -------
    One-hot encoded array
    """
    ncols = a.max()+1
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def ensure_chunks_aligned(arrs:list, axis:int=0, verbose=True) -> list:
    """ 
    Given a list of numpy or dask arrays, ensure that all of the
    dask arrays are chunked along one target axis and have the
    same dimensions

    Parameters
    ----------
    arrs:  
        a list of da.Array or np.ndarray objects
    axis: 
        axis along which chunks must be distinguished

    Returns
    -------
    A list of the arrays with the chunks reconfigured to match
    """
    # identify dask arrays in input sequence
    darrs = [a for a in arrs if isinstance(a, da.Array)]
    if not darrs:
        log.warning("No dask arrays found in input!")
        return arrs
    # check chunks of any dask arrays are split along the target axis
    # (interpreted as training examples), and that they are the same size
    # for now, find one that meets the criterion (assuming present)
    try:
        # numblocks attr refers to the number of blocks present in a given dimension
        # so e.g. if numblocks[-1] == 3, the channel dimension intersects 3 blocks
        # which isnt ideal (we want each single 3D image to be in the same blocks)
        # so all should be 1 except the stacking axis (=0)
        darr_chunky = [
            d for d in darrs if d.numblocks[axis] >= 1 and
            all((d.numblocks[i] == 1 for i in range(len(d.numblocks)) if i != axis))
        ][0]
        target_chunksize = darr_chunky.chunksize
        if verbose:
            log.debug(darr_chunky)
            log.debug(f"Aligning array has numblocks {darr_chunky.numblocks}")
            log.debug(f"Identified target chunksize {target_chunksize}...")
    except IndexError as ie:
        log.error("No dask array found in input sequence with chunks aligned "
                  "along first dimension. Rechunk manually or improve this "
                  "function to do so automatically.")
        raise
    # ensure other dask arrays have the same chunk dimensions (along
    # the target axis and with)
    dsk_inds = []
    for i in range(len(arrs)):
        if isinstance(arrs[i], da.Array):
            dsk_inds.append(i)
            if arrs[i].chunksize != target_chunksize:
                arrs[i] = arrs[i].rechunk(target_chunksize)
    # sanity check
    try:
        assert len(set(arrs[i].chunks for i in dsk_inds)) == 1, (
            "consistency check failed!"
        )
    except AssertionError as a:
        log.debug(f"dsk_inds: {dsk_inds}")
        log.debug(f"chunks: {[arrs[i].chunks for i in dsk_inds]}")
        raise

    return arrs


def resize_array_to_fit_window(arr,
                               window_dims=(224,224),
                               behaviour='crop'):
    """ 
    Resize a dask/np array to a size for which each dimension is
    an integer multiple of window_dims

    Currently assumes windows are extracted in first two dimensions

    Parameters
    ----------
    arr : 
        A numpy/dask array with first two dimensions (rows, cols)
    window_dims :
        pixel (H, W) of window
    """
    assert len(arr.shape) >= 2
    n_rows_window, n_cols_window = window_dims
    assert n_rows_window == n_cols_window, "Use square window dimensions!"
    # calculate number of full window-breadths in each dimension
    n_windows_r, n_windows_c = [int(n) for n in np.array(arr.shape[:2])/n_rows_window]
    # get the part of the array which divides evenly into windows
    _arr_w = arr[:n_windows_r*n_rows_window, :n_windows_c*n_cols_window]
    if behaviour == 'crop':
        return _arr_w
    elif behaviour == 'repeat_edges':
        raise NotImplementedError
    else:
        raise NotImplementedError(f"'{behaviour}' behaviour not implemented!")


def shuffle_array(arr:da.Array, seed:int=42):
    """ 
    Seeded random permutation delegating to faster dask array shuffle method

    Parameters
    ----------
    arr : 
        np/dask array
    seed : 
        integer random seed

    Returns
    -------
    Shuffled array of type same as arr
    """
    random_index_permutation = (
        np.random.RandomState(seed=seed).permutation(arr.shape[0])
    )
    if isinstance(arr, np.ndarray):
        return arr[random_index_permutation]
    elif isinstance(arr, da.Array):
        return da.slicing.shuffle_slice(arr, random_index_permutation)


def split_array(arr, split_fracs:tuple=(0.7, 0.15, 0.15)):
    """ 
    Given an array range in a given dimension, return an N-fold
    partitioning according to some N split_fractions

    Parameters
    ----------
    arr: 
        array
    split_fracs : 
        tuple of floats - fraction of elements in each

    Returns
    -------
    a list of arrays with relative lengths split_fracs
    """
    # sanity check split fractions: partitions should sum to 100%
    assert sum(split_fracs) == 1.0
    split_fracs = np.array(split_fracs)
    # add a zero to the beginning to aid forming index partitions
    if split_fracs[0] != 0.0:
        split_fracs = np.concatenate([[0.0], split_fracs])
    # get a partitioning of these according to split_fracs
    partitioning_ixs = np.cumsum(arr.shape[0]*split_fracs)
    # round and convert to integers for indexing
    partitioning_ixs = np.round(partitioning_ixs).astype(np.int64)
    return [
        arr[i1:i2] for i1, i2 in zip(partitioning_ixs, partitioning_ixs[1:])
    ]


def shuffle_and_split_array(arr, split_fracs=(0.7, 0.15, 0.15), seed=42):
    """ 
    Simple composition of shuffle_array and split_array

    Parameters
    ----------
    arr:
        Dask/numpy array to shuffle and split
    split_fracs:
        The fractions of total length of the original array that each
        of the resulting list of split arrays has
    seed:
        random seed

    Returns
    -------
    A list of shuffled arrays, partitioned into size determined by split_fracs
    """
    return split_array(shuffle_array(arr, seed=seed), split_fracs=split_fracs)


def offset_by_strides(arr:np.array,
                      stride_size:int,
                      n_strides:int) -> list:
    """ 
    From a base array, a stride_size and a number of strides
    create a list of arrays formed by offsetting the original
    by N * stride_size in each of the first two dimensions, where
    N <= n_strides

    Parameters
    ----------
    arr:
        a np/dask array with at least 2 dimensions
    stride_size: 
        an integer number of array indices per stride
    n_stride: 
        the max number of strides

    Returns
    -------
    A list of arrays formed by each of the offsets
    """
    _tiles = []
    # build a list of arrays, each shifted in the first two dimensions
    # by stride * a multiple of n_strides, starting from 0
    for i in range(n_strides):
        for j in range(n_strides):
            _tiles.append(arr[i * stride_size: , j * stride_size:])
    return _tiles


def stride_overlap_and_resize(arr:np.array,
                              window_dims:tuple=(224,224),
                              overlap:float=0.5,
                              behaviour:str='crop'):
    """ 
    From an array, generate a list of arrays formed by offsetting in both the row
    and column dimensions by strides according to an overlap and window_dims
    Resize each of the results to have rows/cols at an integer multiple of window_dims.

    This is used to artificially increase the number of square windows which will
    be the atomic units of training data by creating auxiliary, large arrays from
    which these will be cut out, resulting in a set of overlapping windows.

    Parameters
    ----------
    arr : 
        a numpy/dask array with at least two dimensions
    window_dims :
        tuple of (n_rows, n_cols)
    overlap :
        float - the overlap fraction
    behaviour : 
        method to resize the strided arrays (this is fed to resize_array_to_fit_window)

    Returns
    -------
    a list of resized arrays
    """
    n_rows_window, n_cols_window = window_dims
    assert n_rows_window == n_cols_window, "Use square window dimensions!"
    stride = int((1-overlap) * n_rows_window)
    # we calculate the number of strides we need to shift the whole array
    # by in each dimension to get the correct no. of overlapping windows
    stride_range = int(np.ceil(n_rows_window/stride))
    # derive the offset arrays made by each of the directional strides
    _arr_list = offset_by_strides(arr,
                                  stride_size=stride,
                                  n_strides=stride_range)
    # now resize each of these to fit the windows
    resized_offset_arrs = [
        resize_array_to_fit_window(
            arr,
            window_dims=window_dims,
            behaviour=behaviour
        )
        for arr in _arr_list
    ]
    return resized_offset_arrs




def random_partitioned_index_permutation(ind1,
                                         ind2=None,
                                         step=1,
                                         split_fracs:tuple=(0.7, 0.15, 0.15),
                                         seed=42):
    """ 
    Given an array range in a given dimension, derive an N-fold
    partitioning of indices spanning this range according to some
    split fractions

    Parameters
    ----------
    ind1 : 
        the size of array if ind2 isn't supplied, otherwise the starting index
    ind2 : 
        1 + the largest index to appear
    step : 
        the step in indices, like the third argument of range
    split_fracs : 
        a tuple of floats specifying the fractional partition sizes

    Returns
    -------
    A tuple of shuffled lists of indices, arranged according to split_fracs
    """
    # interpret index arguments like range, specifying the range of
    # indices for which we will generate permutations
    if ind2 is None or ind2 == ind1:
        rng = range(0, ind1, step)
    else:
        rng = range(ind1, ind2, step)
    min_ind, max_ind = min(rng), max(rng)
    size = len(rng)
    # sanity check split fractions: partitions should sum to 100%
    assert sum(split_fracs) == 1.0
    split_fracs = np.array(split_fracs)
    # add a zero to the beginning to aid forming index partitions
    if split_fracs[0] != 0:
        split_fracs = np.concatenate([[0.0], split_fracs])
    # -- firstly, deal with the shared indices
    # get a random permutation of the indices
    rand_ix_perm = np.random.RandomState(seed=seed).permutation(rng) + min_ind
    # get a partitioning of these according to split_fracs
    rand_ix_partitioning = np.cumsum(size*split_fracs)
    # round and convert to integers for indexing
    rand_ix_partitioning = np.round(rand_ix_partitioning).astype(np.int64)
    rand_ixs = [
        rand_ix_perm[i1:i2]
        for i1, i2 in zip(rand_ix_partitioning, rand_ix_partitioning[1:])
    ]
    return rand_ixs

def view_offsets(arr:np.array,
                 window_dims:tuple=(224,224),
                 overlap:float=0.5,
                 behaviour:str='crop'):
    """ 
    Visualise result of offsetting array and tiling it

    Parameters
    ----------
    arr:
        Numpy/dask image array (H, W, C)
    window_dims:
        Tile size
    overlap:
        overlap fraction between windows
    behaviour:
        crop
    """
    # create offset arrays
    d00, d01, d10, d11 = stride_overlap_and_resize(
        arr,
        window_dims=window_dims,
        overlap=overlap,
        behaviour=behaviour
    )
    # look at the first element of each. these should overlap in rows/cols/both
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16,8))
    ax1.imshow(d00[:window_dims[0], :window_dims[1]])
    ax2.imshow(d01[:window_dims[0], :window_dims[1]])
    ax3.imshow(d10[:window_dims[0], :window_dims[1]])
    ax4.imshow(d11[:window_dims[0], :window_dims[1]])
    plt.show()


def cubify(arr:np.array,
           new_shape:tuple):
    """ 
    Rearrange large array into hypertiles, with these distributed
    along a new dimension.

    For example:
        8 x 8 array w/ new_shape (2,2) -> (16,2,2 array)

    Parameters
    ----------
    arr: np.array or dask.array.Array
    new_shape: a tuple, list or iterable specifying the
                shape of the desired hypertiles.

    Returns
    -------
    An array of tiles with axis 0 being a new dimension

    See Also
    --------
    https://stackoverflow.com/questions/42297115/
    numpy-split-cube-into-cubes/42298440#42298440
    """
    old_shape = np.array(arr.shape)
    repeats = (old_shape / new_shape).astype(int)
    tmpshape = np.stack((repeats,new_shape), axis=1)
    tmpshape = tmpshape.ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # new_shape must divide old_shape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(*order).reshape(-1, *new_shape)


def uncubify(arr:np.array,
             old_shape:tuple):
    """ 
    Untile an array, reducing its dimensionality by assembling
    slices together in a grid in other dimensions.

    The inverse of the cubify operation.

    Parameters
    ----------
    arr: np.array or dask.array.Array
    old_shape: a tuple, list or iterable specifying the
                shape of the untiled array.

    Returns
    -------
    An array with one fewer dimension with tiles stitched together

    See Also
    --------
    https://stackoverflow.com/questions/42297115/
    numpy-split-cube-into-cubes/42298440#42298440
    """
    N, new_shape = arr.shape[0], arr.shape[1:]
    old_shape = np.array(old_shape)
    repeats = (old_shape / new_shape).astype(int)
    tmpshape = np.concatenate([repeats, new_shape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(*order).reshape(old_shape)


def test_cubification():
    """ Test the cubify and uncubify operations on dask arrays
    """
    tests = [[da.from_array(np.arange(4*6*16)), (4,6,16), (2,3,4)],
             [da.from_array(np.arange(8*8*8*8)), (8,8,8,8), (2,2,2,2)]]

    for arr, old_shape, new_shape in tests:
        arr = arr.reshape(old_shape)
        assert np.allclose(uncubify(cubify(arr, new_shape), old_shape), arr)
    # cuber = Cubify(old_shape,new_shape)
    # assert np.allclose(cuber.uncubify(cuber.cubify(arr)), arr)


def window_slices(arr,
                  window_dims=(224, 224),
                  overlap=.5) -> (slice, slice):
    """ 
    Generator to yield slices with which to get patches from a large array

    Parameters
    ----------
    arr : 
        an array type structure
    window_dims : 
        the size of the desired windows
    overlap : 
        the fraction that each patch should overlap the next

    Yields
    ------
    (x_slice, y_slice) for each patch in turn
    """
    n_rows_window, n_cols_window = window_dims
    assert n_rows_window == n_cols_window, "Use square window dimensions!"
    stride = int((1-overlap) * n_rows_window)
    # yield n_rows_window*n_cols_window slices
    for row_ix, col_ix in strided_grid(arr, stride):
        # make sure we don't overstep the size of the array
        if row_ix + n_rows_window > arr.shape[0]:
            row_ix = arr.shape[0] - n_rows_window
        if col_ix + n_cols_window > arr.shape[1]:
            col_ix = arr.shape[1] - n_cols_window
        yield (slice(row_ix, row_ix + n_rows_window),
               slice(col_ix, col_ix + n_cols_window))


def count_windows(arr, window_dims=(224, 224), overlap=0.5) -> int:
    """ 
    Count the number of windows in an image, given a window size and overlap

    Parameters
    ----------
    arr : 
        an array type data structure
    window_dims : 
        the size of the desired patches
    overlap :
        the fraction that each window should overlap the next

    Returns
    -------
    integer no. of windows
    """
    n = 0
    stride = int((1-overlap) * window_dims[0])
    for _ in strided_grid(arr, stride):
        n += 1
    return n


def plot_pair(imgs, masks, ix=None, overlay=False, figsize=(16,8)):
    """
    Plot an image/mask pair from a pair of arrays with shape (N, H, W, C(')) 

    Parameters
    ----------
    imgs:
        An array of shape (N, H, W, C) containing many image patches
    masks: 
        An array of shape (N, H, W, C') containing many mask patches
    ix:
        Index of patch to plot. Random if None.
    overlay:
        Flag to superimpose mask on image
    figsize:
        Passed to matplotlib subplots

    Returns
    -------
    matplotlib pyplot figure
    """
    if ix is None:
        ix = np.random.randint(imgs.shape[0])
        log.info("Showing images at index:", ix)
    return plot_img_and_mask(imgs[ix], masks[ix], overlay=True, figsize=(16,8))


def plot_img_and_mask(img, mask, overlay=False, figsize=(16,8), horizontal=True):
    """
    Creates a figure of an image and mask array side-by-side

    Parameters
    ----------
    img:
        Array of shape (H, W, C)
    mask:
        Array of shape (H, W, C')
    overlay:
        Toggles overlaying the mask on the image
    figsize:
        Passed to matplotlib subplots
    horizontal:
        If true, plot side-by-side rather than vertically arranged

    Returns
    -------
    matplotlib figure
    """
    if overlay:
        masked = np.ma.masked_where(mask == 0, mask)
    if horizontal:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    ax1.imshow(img,  interpolation='none')
    ax2.imshow(da.squeeze(mask))
    #if overlay:
    #    ax2.imshow(img, interpolation='none')
    #    ax2.imshow(masked, 'jet', interpolation='none', alpha=0.7)
    plt.show()
    return fig


def plot_sample_segmentation(images:Union[da.Array, np.array],
                             masks:np.array,
                             model:tf.keras.Model,
                             ix:int=None,
                             **kwargs):
    """ 
    Convenience function to plot image/true/predicted segmentation masks
    either randomly (default) or indexed

    Delegates to plot_segmentation after deriving inputs

    Parameters
    ----------
    images : 
        a dask/numpy array containing RGB images
    masks : 
        a dask/numpy binary array of ground truth masks
    model : 
        a tf.keras model with a predict method
    ix : 
        optional integer index

    Returns
    -------
    A matplotlib figure
    """
    # if no index supplied, get a random one
    if ix is None:
        ix = np.random.randint(images.shape[0])
        log.info(f"Showing segmentation mask for image at index {ix}...")
    # get the image and true mask
    img = images[ix]
    if isinstance(images, da.Array):
        img = img.compute()
    msk = masks[ix]
    if isinstance(masks, da.Array):
        msk = msk.compute()
    # predict the mask
    pred_mask = model.predict(img[np.newaxis]/255., steps=1)[0]
    return plot_segmentation(img, msk, pred_mask, **kwargs)


def plot_segmentation(img:np.array,
                      true_mask:np.array=None,
                      pred_mask:np.array=None,
                      threshold:float=None,
                      overlay:bool=False,
                      figsize:tuple=(30,10),
                      return_plots:bool=True) -> None:
    """ 
    Plots an image and optionally also true and predicted segmentation masks,
    with the further possibility of producing another binary segmentation map
    by defining a float threshold

    Parameters
    ----------
    img : 
        a numpy array representing a base RGB image
    true/pred_mask : 
        numpy arrays with the same dimensions as img, but 1 channel
    threshold : 
        a float value to specify the binary decision threshold in pred_mask
    overlay : 
        bool whether to superimpose the true mask on the iamge
    figsize : 
        tuple fig size for matplotlib
    return_plots :
        bool whether to return fig and axes

    Returns
    -------
    optionally fig and axes
    """
    # figure out how many plots we need
    n = len([i for i in (true_mask, pred_mask) if i is not None]) + 1
    if threshold and pred_mask is not None:
        n += 1
    fig, axarr = plt.subplots(1, n, figsize=figsize)
    i = 0
    # base image with optional mask overlay
    axarr[i].imshow(img, interpolation='none')
    axarr[i].set_title('image')
    if overlay and true_mask is not None:
        masked = np.ma.masked_where(np.squeeze(true_mask) == 0, np.squeeze(true_mask))
        axarr[i].imshow(masked, 'jet', interpolation='none', alpha=0.7)
    i += 1
    if true_mask is not None:
        axarr[i].imshow(np.squeeze(true_mask))
        axarr[i].set_title('true mask')
        i += 1
    if pred_mask is not None:
        im = axarr[i].imshow(np.squeeze(pred_mask))
        axarr[i].set_title('predicted mask')
        #divider = make_axes_locatable(axarr[i])
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #fig.colorbar(im, ax=cax)
        i += 1
        if threshold:
            #masked = np.ma.masked_greater_equal(pred_mask, threshold)
            axarr[i].set_title('thresholded mask')
            axarr[i].imshow(np.squeeze(pred_mask) >= threshold)
    plt.show()
    if return_plots:
        return fig, axarr


def yield_chunks(arrs, axis:int=0, loop=False, verbose=False):
    """ 
    Given some iterable of dask/np arrays, each with the same lengths along
    the selected axis, yield numpy arrays for each of these by block
    (for the dask arrays) or by the matching index ranges (for the np arrays)

    Parameters
    ----------
    arrs: 
        a sequence of da.Array/np.ndarray objects which have the same shape
    axis: 
        which axis is understood to align the different arrays for example, 
        along which are the different training examples
    loop: 
        bool whether to continuously proceed

    Yields
    ------
    a tuple of numpy arrays, containing entries from each dask block
    """
    # if special case of just one array rather than iterable, wrap in list
    if isinstance(arrs, np.ndarray) or isinstance(arrs, da.Array):
        arrs = [arrs]
    # check all array lengths are the same along target axis
    assert len(set(a.shape[axis] for a in arrs)) == 1
    # ensure dask array chunks are oriented along axis and identical
    arrs = ensure_chunks_aligned(list(arrs), axis=axis, verbose=verbose)
    # get the indices of the dask arrays
    darr_ixs = [i for i in range(len(arrs)) if isinstance(arrs[i], da.Array)]
    # get dask array to access chunk configuration
    darr = arrs[darr_ixs[0]]
    # get the index pairs along the target axis which make up each chunk
    cumulative_chunk_lens = np.cumsum(np.array(np.concatenate([[0], darr.chunks[axis]])))
    inds = [range(*z) for z in zip(cumulative_chunk_lens, cumulative_chunk_lens[1:])]
    # we iterate over the block number and indices (shared across all darrs)
    while True:
        for i, blk_ind in enumerate(np.ndindex(darr.numblocks)):
            #log.debug(i, blk_ind)
            blk_inds = inds[i]
            to_yield = []
            yield tuple(
                a.blocks[blk_ind].compute() if isinstance(a, da.Array) else
                a[blk_inds]
                for a in arrs
            )
        if not loop:
            break

