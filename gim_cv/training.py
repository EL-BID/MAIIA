"""
training.py

Provides high-level training dataset classes which collect and apply
preprocessing operations to (sets of) rasters/images and provide batch
generators for training models.

Main APIs are the TrainingDataset and CompositeTrainingDataset classes.
"""
import yaml
import random
import pickle
import shutil
import warnings
import logging
import importlib
import hashlib
import itertools
import gc
import abc
from functools import partial
from pathlib import Path, PosixPath
from time import perf_counter as pc
from typing import Union, Any, Tuple

import timbermafia as tm
import pandas as pd
import regex as re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dask
import dask.array as da
from osgeo import gdal, osr, ogr
from cached_property import cached_property

import gim_cv.config as cfg
import gim_cv.preprocessing as preprocessing
from gim_cv.exceptions import InvalidArrayError
from gim_cv.preprocessing import rescale_image_array, balanced_oversample, strong_aug
from gim_cv.interfaces import get_interface
from gim_cv.interfaces.base import ArrayCache
from gim_cv.utils import (yield_chunks, window_slices,
                          count_windows,
                          shuffle_and_split_array, require_attr_true,
                          resize_array_to_fit_window, cubify)
from gim_cv.dask_tools import (stack_interleave_flatten, pair_chunk_generator,
                               shuffuhl_together, get_full_blocks,
                               get_incomplete_blocks)


log = logging.getLogger(__name__)


def from_first_constituent(f):
    """ decorator to mirror properties from first constituent dataset """
    property_name = f.__name__
    def f(self):
        return getattr(self.constituents[0], property_name)
    return f

def pair_batch_generator(imgs:da.Array,
                         masks:da.Array,
                         batch_size:int,
                         img_aug,
                         mask_aug,
                         seed:int=42,
                         shuffle_blocks_every_epoch:bool=True,
                         shuffle_within_blocks:bool=True,
                         axis=0,
                         float32=True):
    """
    Generates batches of image/mask pairs from dask arrays with augmentations.

    Proceeds chunk by chunk through the dask array, generating smaller numpy
    arrays of the appropriate `batch_size` from these as it goes along. Applies
    image and mask augmentations to each of these.

    Parameters
    ----------
    imgs: :obj:`dask.array.Array`
        A dask array of images distributed along axis 0
    masks: :obj:`dask.array.Array`
        A dask array of masks distributed along axis 0
    batch_size: int
        The batch size of the yielded arrays
    img_aug:
        Generator implementing the interface of
        `tensorflow.keras.preprocessing.image.ImageDataGenerator`
    img_aug:
        Generator implementing the interface of
        `tensorflow.keras.preprocessing.image.ImageDataGenerator`
    seed: int
        Random seed
    shuffle: bool
        Flags whether to shuffle within the chunks.
    axis: int
        The axis index along which the training samples are distributed.

    Yields
    ------
    tuple of :obj:`numpy.ndarray`:
        (images, masks) with augmentations applied and shape (batch_size, h, w, c)

    """
    if axis != 0:
        raise NotImplementedError
    n_samples = imgs.shape[axis]
    # keep yielding
    epochs = 0
    while True:
        # get chunk generator for larger-than-memory image and mask arrays
        chunk_gen =  pair_chunk_generator(imgs, masks, shuffle_blocks=shuffle_blocks_every_epoch)
        # for each image/mask chunk in RAM, do a chunk-epoch worth of
        # data-augmented batch generation
        for i, (image_chunk, mask_chunk) in enumerate(chunk_gen):
            log.debug(f"Dask Chunk: {i+1}\n")
            n_samples_chunk = image_chunk.shape[axis]
            n_batches_chunk = n_samples_chunk/batch_size
            # create batch/augmentation generators
            if float32:
                image_chunk = image_chunk.astype('float32')
                mask_chunk = mask_chunk.astype('float32')
            image_generator = img_aug.flow(image_chunk,
                                           batch_size=batch_size,
                                           seed=seed,
                                           shuffle=shuffle_within_blocks)
            mask_generator = mask_aug.flow(mask_chunk,
                                           batch_size=batch_size,
                                           seed=seed,
                                           shuffle=shuffle_within_blocks)
            # generates (img_batch, mask_batch) pairs
            train_generator = zip(image_generator, mask_generator)
            batches = 0 # track batches
            for img_batch, mask_batch in train_generator:
                log.debug(f"Batch: {batches}")
                batches += 1
                yield img_batch, mask_batch
                del img_batch, mask_batch
                gc.collect()
                #model.fit(img_batch, mask_batch)
                if batches >= n_batches_chunk:
                    # we need to break by hand because gen loops indefinitely
                    break
            del image_chunk, mask_chunk
            gc.collect()
        epochs += 1
        #f verbose:
        log.debug(f"Finished generating epoch: {epochs}!\n")


def augment_image_and_mask(img, mask, augger):
    """
    Apply augmentations to an image and mask together using an 
    Albumentations Transformer.
    
    Parameters
    ----------
    img: np.ndarray
        Image with dimensions (H, W, C)
    mask: np.ndarray
        Mask with dimensions (H, W, C')
    augger: :obj:`albumentations.Compose`
        An Albumentations transformer
        
    Returns
    -------
    The augmented image and mask, stacked together for convenience
    in a single array along the channel dimension
    """
    res = augger(image=img, mask=mask)
    img, mask = res['image'], res['mask']
    return np.concatenate([img, mask], axis=-1)


def augment_all(X, y, augger, client):
    """
    Apply image augmentations to a pair of arrays containing 
    images and masks with dimensions (N, H, W, C) and (N, H, W, C')
    
    Parameters
    ----------
    X: :obj:`numpy.ndarray`
        A numpy image array of dimension (N, H, W, C)
    y: :obj:`numpy.ndarray`
        A numpy mask array of dimension (N, H, W, C')
    augger: :obj:`albumentations.Compose`
        An Albumentations transformer
    client: :obj:`distributed.Client`
        A Dask Distributed client for submitting parallel augmentation jobs
    
    Returns
    -------
    (:obj:`np.ndarray`, :obj:`np.ndarray`):
        The augmented image and mask arrays
    """
    aug_fn = partial(augment_image_and_mask, augger=augger)
    futures = client.map(aug_fn, X, y)
    augmented = client.gather(futures)
    try:
        stacked = np.stack(augmented, axis=0)
    except ValueError:
        log.error("pre-aug arrays X, y:")
        log.error(X)
        log.error(y)
        log.error("stack inputs:")
        log.error(augmented)
        raise
    return stacked[:, :, :, :X.shape[-1]], stacked[:, :, :, X.shape[-1]:]


def fancy_batch_generator(X:da.Array,
                          y:da.Array,
                          batch_size:int,
                          augger,
                          client=None,
                          seed:int=42,
                          shuffle_blocks_every_epoch:bool=True,
                          shuffle_within_blocks:bool=True,
                          deep_supervision:bool=False,
                          float32=True):
    """
    Generates batches of image/mask pairs from dask arrays with augmentations.

    Proceeds chunk by chunk through the dask array, generating smaller numpy
    arrays of the appropriate `batch_size` from these as it goes along. Applies
    image and mask augmentations to each of these.

    Parameters
    ----------
    imgs: :obj:`dask.array.Array`
        A dask array of images distributed along axis 0
    masks: :obj:`dask.array.Array`
        A dask array of masks distributed along axis 0
    batch_size: int
        The batch size of the yielded arrays
    augger: :obj:`albumentations.Compose`
        An albumentations composed transformer class
    client: :obj:`distributed.Client`, optional
        Dask distributed client for mapping parallel augmentation jobs
    deep_supervision: bool, optional
        Instead of returning masks, return a list of masks at 
        [8th, 4th, half, native] resolution. for attention pyramid unet.
    seed: int
        Random seed
    shuffle: bool
        Flags whether to shuffle within the chunks.


    Yields
    ------
    tuple of :obj:`numpy.ndarray`:
        (images, masks) with augmentations applied and shape (batch_size, h, w, c)

    """
    # instantiate distributed scheduler if not passed
    if client is None:
        close_client_after = True
        client = Client(processes=False) 
    # track nsamples
    n_samples = X.shape[0]
    # seed numpy
    if seed:
        np.random.seed(seed)
    # keep yielding
    epochs = 0
    # -- loop over whole dask array
    while True:
        # get chunk generator for larger-than-memory image and mask arrays
        chunk_gen =  pair_chunk_generator(X, y, shuffle_blocks=shuffle_blocks_every_epoch)
        # for each image/mask chunk in RAM, do a chunk-epoch worth of
        # data-augmented batch generation
        # -- loop over chunks: these are now numpy arrays
        for i, (image_chunk, mask_chunk) in enumerate(chunk_gen):
            log.debug(f"Dask Chunk: {i+1}\n")
            # get indices of chunk
            n_samples_chunk = image_chunk.shape[0]
            n_batches_chunk = int(n_samples_chunk/batch_size)
            index = np.arange(0, n_samples_chunk)
            # optionally shuffle inplace
            if shuffle_within_blocks:
                np.random.shuffle(index)
            start_point, batches = 0, 0 # track processed samples and batches
            # -- loop over batches in chunk
            while True:
                log.debug(f"Batch: {batches}")
                inds = index[start_point:start_point+batch_size]
                # augment this batch
                Xb, yb = augment_all(
                    image_chunk[inds],
                    mask_chunk[inds],
                    augger,
                    client
                )
                # scale and convert to float32 if necessary
                Xb = (Xb/255.)
                #yb = (yb/255.) ## --- Oops! this was discovered 07/10/20 after most runs...
                if float32:
                    Xb = Xb.astype('float32')
                    yb = yb.astype('float32')
                # necessary for old version of DS with multiple outputs and losses
                if False:#deep_supervision:
                    yb_by_8 = yb[:,::8,::8,:]
                    yb_by_4 = yb[:,::4,::4,:]
                    yb_by_2 = yb[:,::2,::2,:]
                    yb = [yb_by_8, yb_by_4, yb_by_2, yb]
                # try to clean up autocreated client if tf stops iteration
                try:
                    #print(Xb, yb)
                    yield Xb, yb
                except StopIteration:
                    if close_client_after:
                        client.close()
                    raise
                # clean up and increment batch counter
                #del Xb, yb
                #gc.collect()
                start_point += batch_size
                batches += 1
                # stop if we reach the end of the chunk
                if batches >= n_batches_chunk:
                    break
            # -- clean up chunk loop
            #del image_chunk, mask_chunk
        # -- epoch end
        epochs += 1
        log.debug(f"Finished generating epoch: {epochs}!\n")
        

def has_empty_raster(tds:'TrainingDataset') -> bool:
    """
    Returns True if an training dataset image array is all 255s,
    i.e. an empty raster. Loads the array interfaces in order to compute this.
    """
    tds.load_arrays()
    if tds.image_reader.array.min().compute() == 255:
        log.info(f"Identified empty input raster: {tds.image_src}")
        return True
    return False        


def prune_all_black_or_white(X:da.Array, y:da.Array) -> Tuple[da.Array, da.Array]:
    """
    Removes all black/white images and their corresponding masks
    
    Returns the subset of the input image, mask arrays X, y with shapes 
    (patches, h, w, c) wherever X is "empty", for example from masking
    a non-rectangular shape over a raster and dividing it into patches
    
    Parameters
    ----------
    X: :obj:`da.Array`
        Image patch array, shape (patches, h, w, channels)
    y: :obj:`da.Array`
        Mask patch array, shape (patches, h, w, mask_channels)
        
    Returns
    -------
    :obj:`da.Array`, :obj:`da.Array`
        The pruned image/mask arrays
    """
    log.debug("Pruning all black/white image/mask entries...")
    min_pixel_vals = X.min(axis=(1,2)).compute()
    max_pixel_vals = X.max(axis=(1,2)).compute()
    all_white = min_pixel_vals == (255, 255, 255)
    all_black = max_pixel_vals == (0, 0, 0)
    not_all_black = ~(max_pixel_vals == np.array((0, 0, 0))).all(axis=1)
    not_all_white = ~(min_pixel_vals == np.array((255, 255, 255))).all(axis=1)
    return X[not_all_black & not_all_white], y[not_all_black & not_all_white]


class BaseTrainingDataset(tm.Logged, metaclass=abc.ABCMeta):
    """
    Implements common methods for (Composite)TrainingDataset classes.
    See these.
    """
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, ix):
        """
        Shortcut to access (image, mask) arrays anywhere in the dataset.
        """
        return self.X[ix], self.y[ix]

    @property
    def train_val_test_split(self):
        """
        array_like:
            An sequence of floats summing to 1.0, specifying the fractions of the
            data used to defined the training, validation and test sets
        """
        return self._train_val_test_split

    @train_val_test_split.setter
    def train_val_test_split(self, value):
        assert sum(value) == 1.
        self._train_val_test_split = value

    @property
    @abc.abstractmethod
    def X(self):
        pass

    @property
    @abc.abstractmethod
    def y(self):
        pass

    @property
    def split_partition_block_indices(self):
        """
        array_like:
            The indices of the blocks in the dask array containing the training
            data which define the beginning of each section (train, val, test).
            These respect the chunk structure of the underlying arrays, so that
            each partitioning of the data is an integer number of dask chunks.
        """
        return preprocessing.get_partition_indices(
            self.X.numblocks[0],
            self.train_val_test_split
        )
    
    @property
    def split_partition_indices(self):
        return preprocessing.get_partition_indices(
            self.X.shape[0],
            self.train_val_test_split,
            self.batch_size
        )

    @property
    def X_train(self):
        """
        array_like:
            A subset of the Dask array containing the input image arrays used for
            training, with fractional size of the total data fixed by
            train_val_test_split. Shaped as (batches, n_rows, n_cols, channels).
        """
        #start_blk = self.split_partition_block_indices[0]
        #end_blk = self.split_partition_block_indices[1]
        start_ix = self.split_partition_indices[0]
        end_ix = self.split_partition_indices[1]
        #blk_size = self.X.chunksize[0] # assumes axis=0 aligned chunking!
        return self.X[start_ix:end_ix]#start_blk*blk_size:end_blk*blk_size

    @property
    def y_train(self):
        """
        array_like:
            A subset of the Dask array containing the output image arrays used for
            training, with fractional size of the total data fixed by
            train_val_test_split. Shaped as (batches, n_rows, n_cols, channels).
            Typically a segmentation mask.
        """
        #start_blk = self.split_partition_block_indices[0]
        #end_blk = self.split_partition_block_indices[1]
        start_ix = self.split_partition_indices[0]
        end_ix = self.split_partition_indices[1]
        #blk_size = self.y.chunksize[0] # assumes axis=0 aligned chunking!
        return self.y[start_ix:end_ix]#start_blk*blk_size:end_blk*blk_size

    @property
    def X_val(self):
        """
        array_like:
            As X_train, but for validation and with a different fractional size.
        """
        #start_blk = self.split_partition_block_indices[1]
        #end_blk = self.split_partition_block_indices[2]
        start_ix = self.split_partition_indices[1]
        end_ix = self.split_partition_indices[2]
        #blk_size = self.X.chunksize[0] # assumes axis=0 aligned chunking!
        return self.X[start_ix:end_ix]#start_blk*blk_size:end_blk*blk_size]

    @property
    def y_val(self):
        """
        array_like:
            As y_train, but for validation and with a different fractional size.
        """
        #start_blk = self.split_partition_block_indices[1]
        #end_blk = self.split_partition_block_indices[2]
        start_ix = self.split_partition_indices[1]
        end_ix = self.split_partition_indices[2]
        #blk_size = self.y.chunksize[0] # assumes axis=0 aligned chunking!
        return self.y[start_ix:end_ix]#start_blk*blk_size:end_blk*blk_size]

    @property
    def X_test(self):
        """
        array_like:
            As X_train, but for testing and with a different fractional size. If a
            two-fold partitioning of train_val_test_split (e.g. (0.8, 0.2)) is used
            for training and validation only, accessing this will return None.
        """
        if len(self.split_partition_indices) > 3:
            #start_blk = self.split_partition_block_indices[2]
            #end_blk = self.split_partition_block_indices[3]
            start_ix = self.split_partition_indices[2]
            end_ix = self.split_partition_indices[3]
            #blk_size = self.X.chunksize[0] # assumes axis=0 aligned chunking!
            return self.X[start_ix:end_ix]#start_blk*blk_size:end_blk*blk_size]

    @property
    def y_test(self):
        """
        array_like:
            As y_train, but for testing and with a different fractional size. If a
            two-fold partitioning of train_val_test_split (e.g. (0.8, 0.2)) is used
            for training and validation only, accessing this will return None.
        """
        if len(self.split_partition_indices) > 3:
            #start_blk = self.split_partition_block_indices[2]
            #end_blk = self.split_partition_block_indices[3]
            start_ix = self.split_partition_indices[2]
            end_ix = self.split_partition_indices[3]
            #blk_size = self.y.chunksize[0] # assumes axis=0 aligned chunking!
            return self.y[start_blk*blk_size:end_blk*blk_size]

    def batch_gen_train(self):
        """
        Yields
        -------
        :obj:`tuple` of array_like:
            Matching pairs of numpy arrays for each dask chunk for the training
            data as X, y pairs.
        """
        yield from self.batch_generator_fn(self.X_train, self.y_train)

    def batch_gen_val(self):
        """
        Yields
        -------
        :obj:`tuple` of array_like:
            Matching pairs of numpy arrays for each dask chunk for the validation
            data as X, y pairs.
        """
        yield from self.batch_generator_fn(self.X_val, self.y_val)

    def batch_gen_test(self):
        """
        Yields
        -------
        :obj:`tuple` of array_like:
            Matching pairs of numpy arrays for each dask chunk for the testing
            data as X, y pairs.
        Raises
        ------
        ValueError
            If there is no testing fraction (i.e. train_val_test_split has only
            two values, interpreted as training and validation data only).
        """
        if self.X_test is not None:
            yield from self.batch_generator_fn(self.X_test, self.y_test)

    @property
    def tags(self):
        """
        :obj:`set` of :obj:`str`:
            The set of all unique tags associated with the training dataset(s).
        """
        return set(self.tag)


class CompositeTrainingDataset(BaseTrainingDataset, tm.Logged):
    """
    Interface for preparing diverse raster/image data for training.

    Provides consolidated preprocessing, shuffling and batch generation for
    training data spanning multiple rasters/images, each potentially subject to
    its own independent preprocessing operations. Combines multiple independent
    TrainingDatasets and adds (dask) shuffling and caching operations on the
    composite arrays produced by mixing these.

    Allows some flexibility in how batches are produced by setting the
    batch_generator_fn attribute.

    Parameters
    ----------
    constituents : :obj:`list` of `TrainingDataset`
        A list of the individual `TrainingDataset` objects constituting this
        composite dataset, each of which interfaces to a raster file and mask.
    batch_generator_fn: optional
        Generator function with signature (X, y), where these are the training
        inputs and masks respectively. Should return (x', y') numpy pairs for
        each batch, each with shape (batch_size, H, W, C). Defaults to using the
        same function used to generate batches as the first constituent
        TrainingDataset.
    cache_directory: :obj:`str` or :obj:`pathlib.Path`, optional
        Path pointing to a directory where the processed X and y training arrays
        can be dumped for re-use.
    oversample_fn: callable, optional
        A function with signature (image_arrays, mask_arrays), where each 
        is a list of arrays originating from different datasets. If this is 
        not None, it will be called early in the combination of different 
        TrainingDatasets to ensure equal representation of training examples
        from each dataset. Should return a tuple the same form as its 
        signature: (image_arrays, masked_arrays), where these are assumed to 
        now be oversampled.
    prune_fn: callable, optional
        A callable which accepts a TrainingDataset, performs some calculation
        and returns True if it's to be removed from the constituents else False.
        Used for eliminating rasters which are e.g. empty (all 255s).

    Notes
    -----

    Implements __add__ so that one can produce CompositeTrainingDatasets by
    adding TrainingDatasets or CompositeTrainingDatasets to each other.

    Some properties are copied from the first consituent TrainingDataset by
    convention for the purposes of shuffling and rebatching, for example the
    batch_size, seed and train_val_test_split.
    """

    def __init__(self,
                 constituents=[],
                 batch_generator_fn=None,
                 cache_directory=None,
                 oversample_fn=None,
                 prune_fn=has_empty_raster):
        self.constituents = list(constituents)
        self.batch_generator_fn = None
        self.cache_directory = cache_directory
        self.composition_prepared = False
        self.oversample_fn = oversample_fn
        self.prune_fn = prune_fn
        
    def __add__(self, other):
        if isinstance(other, TrainingDataset):
            self.constituents.append(other)
            self.composition_prepared = False
        elif isinstance(other, CompositeTrainingDataset):
            self.constituents.extend([o for o in other.constituents])
            self.composition_prepared = False
        else:
            raise ValueError(f"Type {type(other)} not understood in "
                             f"constructing composite training dataset")
        return self

    
    @property
    def composition_prepared(self):
        return self._composition_prepared
    
    @composition_prepared.setter
    def composition_prepared(self, value):
        self._composition_prepared = value
    
    @property
    def cache_directory(self):
        return self._cache_directory

    @cache_directory.setter
    def cache_directory(self, value):
        self._cache_directory = value

    @property
    @from_first_constituent
    def seed(self):
        pass

    @property
    @from_first_constituent
    def batch_size(self):
        pass

    @property
    @from_first_constituent
    def train_val_test_split(self):
        pass
    
    @train_val_test_split.setter
    def train_val_test_split(self, value):
        return BaseTrainingDataset.train_val_test_split.setter(self, value)

    @property
    def tags(self):
        return set(c.tag for c in self.constituents if c.tag is not None)

    @property
    def tags_str(self):
        """
        str: A string ID, formed by sorting and concatenating the unique
        TrainingDataset tags
        """
        return '_'.join(t for t in sorted(self.tags))

    def constituent_datasets_by_tag(self, tag):
        """

        Parameters
        ----------
        tag: str
            A string tag associated with a given (set of) dataset(s)

        Returns
        -------
        :obj:`list` of TrainingDataset
            A list of the constituent TrainingDataset objects with this tag
        """
        return [ds for ds in self.constituents if ds.tag == tag]

    @property
    def batch_generator_fn(self):
        return self._batch_generator_fn

    @batch_generator_fn.setter
    def batch_generator_fn(self, fn):
        # grab from first constituent if not explicitly set
        if fn is None:
            self._batch_generator_fn = self.constituents[0].batch_generator_fn
        else:
            self._batch_generator_fn = fn

    @property
    def prepared(self):
        """
        A flag for marking when all preprocessing and batch assembly operations
        have been applied. Gates other methods which require these steps to be
        completed.

        bool:
            True if all consituents have been independently preprocessed and the
            composition into shuffled batches has been completed. Basically if
            the prepare method has been run.
        """
        return (all(c.prepared for c in self.valid_constituents) and
            self.composition_prepared)

    @property
    def valid_constituents(self):
        """
        A flag for marking whether all constituent TrainingDatasets are valid.

        bool:
            True if all constituents' valid attributes are True else False
        """
        return [c for c in self.constituents if c.valid]

    def prune(self):
        """
        Calls prune_fn (if set) on each constituent dataset and removes it from the 
        constituents list if this function returns True.
        """
        if self.prune_fn is not None:
            self.log.info(
                f"Selecting training datasets to eliminate with {self.prune_fn.__name__}..."
            )
            nc = len(self.constituents)
            self.constituents = [c for c in self.constituents if not self.prune_fn(c)]
            self.log.info(
                f"Removed {nc-len(self.constituents)} TrainingDatasets, leaving {nc}."
            )
                          
        else:
            self.log.warning("Prune called but no prune_fn attribute set. Doing nothing.")
    
    def prepare(self, validate=False, shuffle_repeats=4):
        """
        Generates preprocessed image/mask arrays ready for training

        Applies preprocessing, concatenation, chunking, and shuffling operations
        by applying the independent preprocessing operations of each constituent
        TrainingDataset first, then assembling these individual arrays into
        consolidated ones.

        Parameters
        ----------
        validate: bool, optional
            Flag whether to apply array validation (slower)
        shuffle_repeats: int, optional
            Specifies the number of times to interleave slices of each block.

        Notes
        -----
        Due to dask working on blocks of arrays read lazily from multiple files,
        "true" shuffling, which would mix patches from random areas of random
        image files is somewhat tricky as this is very slow and memory intensive.
        Here we use an approximation that interleaves elements from a fixed set
        of random blocks from each source image from each batch. The degree to
        which this kind of shuffling occurs is fixed by shuffle_repeats.

        We see empirically that loss curves in training models smooth out after
        a few iterations of shuffle_repeats and training converges.
        
        This should probably be refactored so this function isn't so monolithic,
        perhaps into a compose_arrays method.
        """
        # remove training datasets by pruning if set
        if self.prune_fn is not None:
            self.prune()
        # prepare constituents (apply pipelines etc)
        for c in self.constituents:
            if not c.prepared:
                # chunks get optimised after combination!
                c.prepare(optimise_chunks=False, validate=validate)
        # -- prepare composition
        # shuffle examples from diff datasets (if they're the same size)
        # build arrays of shuffled samples from each training dataset
        self._X_arrs, self._y_arrs = [], []
        # ignore invalid datasets
        # concatenate together arrays of the same size within each dataset
        # (defined by a tag)
        # replacing stack_interleave_flatten here with plain old concatenate
        # put all the shuffling at the end and keep the distinction of a list of 
        # arrays, one for each tag to allow oversampling
        for t in self.tags:
            constituents = [c for c in self.valid_constituents if c.tag == t]
            self._X_arrs.append(da.concatenate([c.X for c in constituents], axis=0))
            self._y_arrs.append(da.concatenate([c.y for c in constituents], axis=0))
        # treat tagless datasets as being from distinct sources
        self._X_arrs.extend([c.X for c in self.valid_constituents if c.tag is None])
        self._y_arrs.extend([c.y for c in self.valid_constituents if c.tag is None])
        # inject oversample here
        if self.oversample_fn is not None:
            self.log.info(
                "Performing oversampling of arrays from different training datasets "
                f"with function: {self.oversample_fn.__name__}"
            )
            self._X_arrs, self._y_arrs = self.oversample_fn(self._X_arrs, self._y_arrs)
        # reset chunks to ideal size along zeroth dimension after interleaving
        self._X_arrs[0] = self._X_arrs[0].rechunk(('auto', -1, -1, -1))
        ideal_chunksize = self._X_arrs[0].chunksize[0]
        # rechunking and aligning chunks with N batches
        self.log.info("Aligning image/mask array chunks...")
        aligner = preprocessing.ChunkBatchAligner(self.batch_size, ideal_chunksize)
        for ix in range(len(self._X_arrs)):
            self._X_arrs[ix] = aligner.fit_transform(self._X_arrs[ix])
            self._y_arrs[ix] = aligner.transform(self._y_arrs[ix])
        # now interleave data blocks on different datasets over
        # the smallest common number of blocks, sticking the rest
        # on the end
        if len(self._X_arrs) == 1:
            self._X = self._X_arrs[0]
            self._y = self._y_arrs[0]
        elif len(self._X_arrs) > 1:
            self._X = da.concatenate(self._X_arrs)
            self._y = da.concatenate(self._y_arrs)
        # finally fold arrays on themselves, flatten, shuffle the blocks and repeat
        # a few times
        self.log.info("Shuffling datasets together...")
        self._X, self._y = shuffuhl_together(
            self._X, self._y,
            shuffle_blocks=True,
            repeats=shuffle_repeats
        )
        # make sure incomplete blocks are at the end
        X_full_blocks, X_incomplete_blocks = (
            get_full_blocks(self._X), get_incomplete_blocks(self._X)
        )
        y_full_blocks, y_incomplete_blocks = (
            get_full_blocks(self._y), get_incomplete_blocks(self._y)
        )
        self._X = X_full_blocks
        self._y = y_full_blocks
        if X_incomplete_blocks is not None:
            self._X = da.concatenate([self._X, X_incomplete_blocks])
            self._y = da.concatenate([self._y, y_incomplete_blocks])
        self.composition_prepared = True
        self.log.info("Composite dataset prepare done!")

    @property
    def _ca_X(self):
        suffix = f'oversample_{self.oversample_fn.__name__}' if self.oversample_fn is not None else ''
        return ArrayCache(
            array=None,
            cache_path=self.cache_directory / Path(f'X_processed{suffix}')
        )

    @property
    def _ca_y(self):
        suffix = f'oversample_{self.oversample_fn.__name__}' if self.oversample_fn is not None else ''
        return ArrayCache(
            array=None,
            cache_path=self.cache_directory / Path(f'y_processed{suffix}')
        )

    def save_prepared_arrays(self, overwrite=False):
        """
        Saves preprocessed and shuffled training arrays to .zarr format

        The target directory is controlled by the cache_directory attribute.

        Parameters
        ----------
        overwrite: bool
            Flag whether to overwrite existing arrays from the same dataset in
            the same directory
        """
        cX, cy = self._ca_X, self._ca_y
        cX.array, cy.array = self.X, self.y
        cX.save(overwrite=overwrite)
        cy.save(overwrite=overwrite)

    def delete_prepared_arrays(self):
        """
        Deletes any cached zarr arrays generated from save_prepared_arrays
        """
        self._ca_X.delete()
        self._ca_y.delete()

    def load_prepared_arrays(self):
        """
        Loads previously-calculated preprocessed and shuffled training arrays

        The source directory is controlled by the cache_directory attribute.
        """
        self._X = self._ca_X.read()
        self._y = self._ca_y.read()
        # manually override prepared guard variable
        for c in self.constituents:
            c.prepared = True
        self.composition_prepared = True

    @require_attr_true('prepared')
    def get_X(self):
        return self._X

    @require_attr_true('prepared')
    def get_y(self):
        return self._y

    @property
    def X(self):
        """
        Returns the processed input image array.

        The prepare method must previously have been run to access this.

        array_like:
            A consolidated (n_samples, H, W, C) input array for training
        """
        return self.get_X()

    @property
    def y(self):
        """
        Returns the processed output mask array.

        The prepare method must previously have been run to access this.

        array_like:
            A consolidated (n_samples, H, W, C) truth array for training
        """
        return self.get_y()


class TrainingDataset(BaseTrainingDataset, tm.Logged):
    """
    Interface for preparing raster/image data from a file for training.

    Provides consolidated preprocessing, shuffling and batch generation for
    training data spanning multiple rasters/images, each potentially subject to
    its own independent preprocessing operations. Combines multiple independent
    TrainingDatasets and adds (dask) shuffling and caching operations on the
    composite arrays produced by mixing these.

    Allows some flexibility in how batches are produced by setting the
    batch_generator_fn attribute.

    Parameters
    ----------
    image_src : str or :obj:`pathlib.Path`
        Path to a raster/image file which will be moulded into input training
        examples
    mask_src : str or :obj:`pathlib.Path`
        Path to a raster/image file which will be moulded into the corresponding
        training ground truth labels to image_src
    image_pipeline_factory :
        A function which returns a `sklearn.pipeline.Pipeline` which performs
        the preprocessing operations necessary to transform the raw image/raster
        array from image_src into training-ready form.
    mask_pipeline_factory :
        A function which returns a `sklearn.pipeline.Pipeline` which performs
        the preprocessing operations necessary to transform the raw image/raster
        mask/ground truth array from mask_src into training-ready form.
    batch_generator_fn: optional
        Generator function with signature (X, y), where these are the training
        inputs and masks respectively. Should return (x', y') numpy pairs for
        each batch, each with shape (batch_size, H, W, C).
    batch_size : int, optional
        The batch size to be used in training. This will be used to optimise the
        dask chunk sizes.
    image_validation_function : optional
        An optional function to be applied to the raw input image/raster array
        which returns True if it is considered valid. For example, a function
        which returns True only if there are nonzero entries.
    mask_validation_function : optional
        An optional function to be applied to the raw ground truth mask array
        which returns True if it is considered valid. For example, a function
        which returns True only if there are nonzero entries.
    prune_patch_fn : optional
        An optional callable with signature fn(img_array, mask_array) that
        retunrs pruned_img_array, pruned_mask_array
    seed : int
        Random seed for shuffling
    train_val_test_split : array_like
        A sequence of floats summing to 1.0, specifying the fractional sizes of
        the training, validation and testing datasets which the input images and
        masks will be partitioned into.
    tag : str, optional
        A string identifier for this raster dataset. TrainingDatasets possessing
        the same tag will be considered part of the same mother dataset and
        bundled together for e.g. shuffling operations. Different rasters can
        be grouped by similarity of content using this. Used downstream.


    Examples
    --------
    >> from functools import partial
    >> from gim_cv.preprocessing import (
    >>    get_image_training_pipeline, get_binary_mask_training_pipeline,
    >>    get_aug_datagen
    >> )
    >> from gim_cv.training import TrainingDataset, pair_batch_generator
    >>
    >>
    >> # get function to generate batches of images and masks for each dataset,
    >> # choosing a batch size and augmentation generator
    >> batch_generator = partial(pair_batch_generator,
    >>                           batch_size=batch_size,
    >>                           img_aug=get_aug_datagen(),
    >>                           mask_aug=get_aug_datagen(),
    >>                           seed=seed,
    >>                           shuffle=False)
    >>
    >> tds = TrainingDataset(
    >>    './training_data/image_1.tif',
    >>    './training_data/mask_1.tif',
    >>    image_pipeline_factory=get_image_training_pipeline,
    >>    mask_pipeline_factory=get_binary_mask_training_pipeline,
    >>    batch_generator_fn=batch_generator,
    >> )
    >>
    >> # build preprocessing operations
    >> tds.prepare() # now arrays are generated as the X, y attributes
    >>
    >> model.fit_generator(tds.batch_gen_train(),
    >>                     validation_data=training_data.batch_gen_val(),
    >>                     ...)

    Notes
    -----
    Implements __add__ so that one can produce CompositeTrainingDatasets by
    adding TrainingDatasets or CompositeTrainingDatasets to each other.

    """
    prepared = False # default state: require prepare method to be run, loading + preprocessing
    tile_axis = 0    # stack training images as tiles along first dimension

    def __init__(self,
                 image_src,
                 mask_src,
                 image_pipeline_factory,
                 mask_pipeline_factory,
                 batch_generator_fn,
                 batch_size=None, # can optionally ensure chunk size = N *  batch_size
                 image_validation_function=None,
                 mask_validation_function=None,
                 prune_patch_fn=prune_all_black_or_white,
                 seed=42,
                 train_val_test_split=(0.6, 0.2, 0.2),
                 tag=None):
        # if given iterable sof associated image/shapes as arguments, construct
        # a TrainingDataset object for each pair independently then combine them
        try:
            self.image_src = Path(image_src)
            self.mask_src = Path(mask_src)
        except TypeError:
            self.log.debug(
                "Attempting to interpret image/shape arguments as matching "
                "iterables"
            )
            return sum(TrainingData(i, s) for i, s in zip(image, shape))
        self.train_val_test_split = train_val_test_split
        # dispatch maskreader to shapes to produce arrays if necessary
        _im_reader_cls = get_interface(self.image_src, 'ImageReader')
        _msk_reader_cls = get_interface(self.mask_src, 'BinaryMaskReader')
        # instantiate image reader
        self.image_reader = _im_reader_cls(self.image_src, read_metadata=True)
        # instantiate mask reader
        # pass the metadata of the image too incase it needs it (will overwrite if present in tif etc)
        # (e.g. a shapefile requires georeferencing/transform metadata to produce a mask)
        self.mask_reader = _msk_reader_cls(self.mask_src, metadata=self.image_reader.metadata)
        # get references to the pipeline functions
        self.image_pipeline_factory = image_pipeline_factory
        self.mask_pipeline_factory = mask_pipeline_factory
        # get function to generate batches
        self.batch_generator_fn = batch_generator_fn
        # rng
        self.seed = seed
        self.batch_size = batch_size
        # define tag for grouping datasets at a later stage
        self.tag = tag
        # overriden functions to validate inputs
        self.image_validation_function = image_validation_function
        self.mask_validation_function = mask_validation_function
        # flag for if inputs are invalid
        self.valid = True
        self.prune_patch_fn = prune_patch_fn

    def __add__(self, other):
        """ If more than one, return a composite training dataset
        """
        if isinstance(other, TrainingDataset):
            return CompositeTrainingDataset(constituents=[self, other])
        elif isinstance(other, CompositeTrainingDataset):
            return other + self
        else:
            raise ValueError(f"Type {type(other)} not understood in "
                             f"constructing composite training dataset")

    def load_arrays(self, reload=False):
        self.log.debug(f"loading arrays")
        if self.image_reader.array is None or reload:
            self.image_reader.load_array()
        if self.mask_reader.array is None or reload:
            self.mask_reader.load_array()
        self.log.debug(f"done loading arrays")

    def validate_arrays(self):
        self.log.debug(f"validating arrays")
        self.image_reader.validate_array(self.image_validation_function)
        self.mask_reader.validate_array(self.mask_validation_function)
        self.log.debug(f"done validating arrays")

    def make_pipelines(self):
        if self.image_pipeline_factory is None:
            self.image_pipeline = None
        else:
            self.log.debug("Assigning image pipeline...")
            self.image_pipeline = self.image_pipeline_factory()
        if self.mask_pipeline_factory is None:
            self.mask_pipeline = None
        else:
            self.log.debug("Assigning mask pipeline...")
            self.mask_pipeline = self.mask_pipeline_factory()

    def prepare(self, optimise_chunks=True, validate=False):
        """
        Method to preprocess raw raster array data into training-ready state

        Loads image/mask arrays (with cache-reliance behaviour deferred to config
        via use_cache attr of readers), builds preprocessing pipelines then
        applies their fit_transform method to the image/mask arrays to produce
        training arrays with the shape (samples, H, W, C). Performs some batch/
        chunk alignment optimisations and shuffling.

        Parameters
        ----------
        optimise_chunks: bool, optional
            Flag to optimise dask chunks to span an integer multiple of batches
        validate: bool, optional
            Flag to enforce array validation (via image_validation_function
            and mask_validation_function methods) first.

        Notes
        -----
        Can skip pipeline step by explicitly setting pipeline = None
        """
        self.log.debug(f"Obtaining arrays for {self.image_src}, {self.mask_src}...")
        self.load_arrays()
        # validate is run already if arrays are cached
        try:
            if validate:
                self.validate_arrays()
        except InvalidArrayError:
            self.log.error("Invalid array or mask detected, skipping dataset...")
            self.valid=False
            return
        _X, _y = self.image_reader.array, self.mask_reader.array
        # need spatial chunks to align for resampling pixel losses
        if _X.chunks[:2] != _y.chunks[:2]:
            self.log.debug(f"Aligning spatial chunks...")
            _y = _y.rechunk((_X.chunks[0], _X.chunks[1], *_y.chunks[2:]))
        self._X, self._y = _X, _y
        self.make_pipelines()
        if self.image_pipeline is not None:
            self._X = self.image_pipeline.fit_transform(self._X)
            self.log.info("Image pipeline done!")
        if self.mask_pipeline is not None:
            self._y = self.mask_pipeline.fit_transform(self._y)
            self.log.info("Mask pipeline done!")
        assert self._X.shape[0] == self._y.shape[0], "Different numbers of images/masks!"
        # apply pruning here before optimising chunks
        if self.prune_patch_fn is not None:
            self._X, self._y = self.prune_patch_fn(self._X, self._y)
        self.log.debug("Aligning image/mask array chunks...")
        self._X = self._X.rechunk(('auto', -1, -1, -1))
        self._y = self._y.rechunk((self._X.chunks[0], -1, -1, -1))
        if optimise_chunks and self.batch_size is not None:
            self.log.debug("Optimising chunks to factorise into batches...")
            aligner = preprocessing.ChunkBatchAligner(self.batch_size)
            self._X = aligner.fit_transform(self._X)
            self._y = aligner.transform(self._y)
        if self._X.size == 0 or self._y.size == 0:
            raise ValueError("empty training arrays detected! handle this better in composites?")
        self.prepared = True
        self.log.info(f"Image/mask arrays from {self.image_src} <-> {self.mask_src} prepared for training!")

    @property
    def data_uid(self):
        """
        Generates a stable id associated with this dataset's input filenames.
        """
        m = hashlib.md5()
        for s in itertools.chain(self.image_src, self.mask_src):
            m.update(s.encode('utf-8'))
        return m.hexdigest()
        #return tuple_hash_pos_def((self.image_src, self.mask_src))

    @require_attr_true('prepared')
    def get_X(self):
        return self._X

    @require_attr_true('prepared')
    def get_y(self):
        return self._y

    @property
    def X(self):
        """ processed input image array """
        return self.get_X()

    @property
    def y(self):
        """ processed output mask array """
        return self.get_y()

        
# utility functions and classes for loading/saving model checkpoints and suchlike

def get_run_data(model_dir,
                 cp_ptn:str=(
                     'cp-e(?P<epoch>\d+)(-ji(?P<ji>[\d.]+))?-l(?P<loss>[\d.]+)'
                     '-vl(?P<val_loss>[\d.]+).ckpt.index'
                 ),
                 dataset_aliases=None):
    """
    Parse a model directory name, possibly containing checkpoints, to 
    reconstruct the loss function, training data used and best loss achieved
    
    Assumes cp_ptn is a regex containing epoch, loss and val_loss match groups.
    
    Returns
    -------
    :obj:`pd.DataFrame`
        A dataframe containing training metadata such as loss, dataset used, spatial 
        resolution and the paths to the best checkpoints
    """
    uuid4_ = re.match('.*_(?P<uuid4>(\w+-){,4}\w+)$', str(model_dir.parts[-1])).group('uuid4')
    ckpts = model_dir.glob('*.ckpt.index')
    base_paths, losses, val_losses, epochs = [], [], [], []
    for ckpt in ckpts:
        base_paths.append(str(ckpt)[:-6]) # remove index
        cp_match = re.match(
            cp_ptn,
            str(ckpt.parts[-1])
        )
        epochs.append(cp_match.group('epoch'))
        losses.append(cp_match.group('loss'))
        val_losses.append(cp_match.group('val_loss'))
    best_trn_loss_ix = np.argmin(losses) if losses else 0
    best_val_loss_ix = np.argmin(val_losses) if val_losses else 0
    try:
        low_l_cp = base_paths[best_trn_loss_ix]
        low_l = losses[best_trn_loss_ix]
    except IndexError:
        low_l_cp = None
        low_l = None
    try:
        low_vl_cp = base_paths[best_val_loss_ix]
        low_vl = losses[best_val_loss_ix]
    except IndexError:
        low_vl_cp = None
        low_vl = None
    try:
        X0_train = list(model_dir.glob('X_train*.npy'))[0]
        y0_train = list(model_dir.glob('y_train*.npy'))[0]
    except IndexError:
        X0_train = None
        y0_train = None
    try:
        X0_val = list(model_dir.glob('X_val*.npy'))[0]
        y0_val = list(model_dir.glob('y_val*.npy'))[0]
    except IndexError:
        X0_val = None
        y0_val = None
    metadata = {
        'uuid4' : uuid4_,
        'training_dir' : str(model_dir),
        'lowest_loss_ckpt' : low_l_cp,
        'lowest_val_loss_ckpt' : low_vl_cp,
        'lowest_loss' : np.float32(low_l),
        'lowest_val_loss' : np.float32(low_vl),
        'X0_train' : X0_train,
        'y0_train' : y0_train,
        'X0_val' : X0_val,
        'y0_val' : y0_val
    }
    with open(model_dir / Path('run_params.yml')) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    metadata.update(params)
    if dataset_aliases:
        metadata['datasets_alias'] = dataset_aliases.get(metadata['datasets'], metadata['datasets'])
    return pd.Series(metadata)


def collate_run_data(
    models_dir,
    model_name='DeepResUNet',
    dataset_aliases=None
):
    """
    Creates a dataframe containing metadata on all trained models in a directory
    
    model_name can be a regex
    """
    runs = []
    for m in models_dir.glob(f'{model_name}_*'):
        if re.match(f'{model_name}''_(\w+-){,4}\w+$',str(m.parts[-1])):
            try:
                runs.append(get_run_data(m, dataset_aliases=dataset_aliases))
            except Exception as e:
                print(f"{m} failed with exception:")
                print(e)
    return pd.concat(runs, axis=1).T
