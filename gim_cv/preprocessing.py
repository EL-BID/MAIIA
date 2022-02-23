"""
preprocessing.py

Provides preprocessing transformations conforming to the sklearn
Transformer API and auxiliary functions for preparing raster/image data for
machine learning operations.

Main high-level interfaces are currently training and inference pipelines for
semantic segmentation (see `get_image_training_pipeline`), along with
generators for image augmentation (see e.g. `get_fancy_aug_datagen`).
"""
import random

import numpy as np
import dask.array as da
import rasterio
import timbermafia as tm
import cv2
import albumentations as A

from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline#, FeatureUnion#, make_pipeline
from sklearn.preprocessing import FunctionTransformer#, minmax_scale, scale, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.preprocessing import image


import logging

log = logging.getLogger(__name__)

#from ImageDataAugmentor.image_data_augmentor import *
from albumentations.core.transforms_interface import ImageOnlyTransform

RESAMPLE_TOLERANCE = 0.01 # don't resample if spatial resolutions are within 1%


def strong_aug(
    p=.8,
    fancy_pca=None,
    p_rr90=1.,
    p_hflip=.5,
    p_rgb_shift=.4,
    p_iaa_affine=.01,
    p_noise=.2,
    p_blur=.02,
    p_ssr=.2,
    p_distort=.02,
    p_sharpen=0.2,
    p_contrast=0.4,
    p_brightness=0.3,
    p_hue_sat=0.05,
    p_gamma=.1
):
    """ 
    Returns a composition of albumentations image/mask transformations
    
    Parameters
    ----------
    p : float, optional
        Global augmentation probability in [0,1.] (of doing any augs at all)
    fancy_pca : :obj:`FancyPCA`, optional
        An instance of FancyPCA, evaluated on the training dataset 
        with precalculated shift eigenvectors
    p_rr90 : 
        Probability of random rotation by multiple of 90 degrees
    p_hflip : 
        Horizontal flip probability
    p_rgb_shift : 
        Probability of random RGB shift
    p_iaa_affine : 
        Probability of affine transformation
    p_noise : 
        Probability of gaussian noise
    p_blur : 
        Probability of gaussian blur
    p_ssr : 
        Probability of shift-scale-rotate
    p_distort : 
        Probability of distortion
    p_sharpen : 
        Probability of sharpening
    p_contrast : 
        Probability of contrast shift
    p_brightness : 
        Probability of brightness shift
    p_hue_sat : 
        Probability of hue-saturation shift
    p_gamma : 
        Probability of gamma shift
        
    Returns
    -------
    albumentations.core.composition.Compose
        A composition of various albumentations transformations for 
        image augmentation
    """
    aug_list = []
    if p_rr90:
        aug_list.append(A.RandomRotate90(p=p_rr90))
    if p_hflip:
        aug_list.append(A.HorizontalFlip(p=p_hflip))
    if fancy_pca is not None:
        aug_list.append(fancy_pca)
    if p_rgb_shift:
        aug_list.append(A.RGBShift(p=p_rgb_shift,
                       r_shift_limit=(-15, 15),
                       g_shift_limit=(-15, 15),
                       b_shift_limit=(-15, 15)))
    if p_iaa_affine:
        aug_list.append(
            A.IAAAffine(scale=1.0, translate_percent=(-0.1, 0.1), translate_px=None, rotate=0.0, shear=(-10, 10),
                        order=1, cval=0, mode='reflect', always_apply=False, p=p_iaa_affine)
        )
    if p_noise:
        aug_list.append(
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(var_limit=(5., 20.)),
            ], p=p_noise)
        )
    if p_blur:
        aug_list.append(
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 4), p=1.),
                #A.MedianBlur(blur_limit=(3, 5), p=.1),
                A.Blur(blur_limit=(3,4), p=1.),
            ], p=p_blur)
        )
    if p_ssr:
        aug_list.append(
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=p_ssr)
        )
    if p_distort:
        aug_list.append(
            A.OneOf([
                #A.OpticalDistortion(p=0.3),
                #A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=1.0),
            ], p=p_distort)
        )
    if p_sharpen:
        aug_list.append(A.IAASharpen(p=p_sharpen))
    if p_contrast:
        aug_list.append(
            A.OneOf([
                A.CLAHE(clip_limit=2),
                #A.IAAEmboss(),
                A.RandomContrast(),
            ], p=p_contrast)
        )
    if p_brightness:
        aug_list.append(A.RandomBrightness(p=p_brightness))
    if p_hue_sat:
        aug_list.append(
            A.HueSaturationValue(10,10,10,p=p_hue_sat)
        )
    if p_gamma:
        aug_list.append(
            A.RandomGamma(gamma_limit=(90, 110), p=p_gamma)
        )
    return A.Compose(aug_list, p=p)


def get_aug_datagen(rescale=1/255.,
                    featurewise_center=False, # *requires implementing fit method for dask
                    featurewise_std_normalization=False, # *
                    zca_whitening=False, # *
                    horizontal_flip=False,
                    vertical_flip=False,
                    rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    zoom_range=0.0):
    """
    Factory returning a data generator for image/mask augmentation.

    Returns a tf.keras.preprocessing.image.ImageDataGenerator, which applies
    transformations to numpy arrays during training.

    Default just rescales RGB by 1/255.

    Parameters
    ----------
    rescale : float, optional
        Rescale factor for 8-bit integer RGBs to floats, typically 1/255.
    featurewise_center : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    featurewise_std_normalization : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    zca_whitening : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    horizontal_flip : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    vertical_flip : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    rotation_range : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    width_shift_range : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    height_shift_range : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator
    zoom_range : bool, optional
        See tf.keras.preprocessing.image.ImageDataGenerator

    Returns
    -------
    tf.keras.preprocessing.image.ImageDataGenerator
        An image data generator with the requested parameters
    """
    return image.ImageDataGenerator(**locals())


def get_image_inference_pipeline(inference_window_size=896,
                                 resample_factor=1.0,
                                 preserve_int=True,
                                 resample_tolerance=RESAMPLE_TOLERANCE,
                                 inference_batch_size=1,
                                 channels=[0,1,2]):
    """
    Factory returning a pipeline to preprocess raster data for inference.

    Applies operations for selecting, padding, resampling, tiling and
    rescaling raw RGB dask arrays extracted from rasters into batches ready for
    inference

    Arguments
    ---------
    inference_window_size: array_like
        The desired row, col size of patches used for inference. Usually limited
        by GPU memory (for example, (896,896))
    resample_factor: float
        see ImageResampler
    preserve_int: bool
        see ImageResampler
    resample_tolerance: float
        see ImageResampler
    inference_batch_size: int
        The number of patches per batch for inference. Defaults to 1, as this
        allows the largest possible patch size thus mitigating edge-artifacts in
        the output of inference.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline encapsulating the preprocessing stages
    """
    return Pipeline(steps=[
        ('channel_selector', ChannelSelector(channels=list(channels))),
        ('array_padder', ArrayPadder(inference_window_size,
                                     inference_window_size)),
        ('resampler', ImageResampler(sf=resample_factor,
                                     preserve_int=preserve_int,
                                     resample_tolerance=resample_tolerance)),
        ('tiler', Tiler(window_dims=(inference_window_size, inference_window_size))),
        ('rechunker', Rechunker((inference_batch_size, -1, -1, -1))),
        ('scaler', SimpleInputScaler(1/255.)),
        ('float32er', Float32er())
    ])


def get_image_training_pipeline(window_dims=(256,256),
                                resample_factor=1.0,
                                overlap_tiles=True,
                                preserve_int=True,
                                seed=42,
                                channels=[0,1,2],
                                prepend_steps=[],
                                resample_tolerance=RESAMPLE_TOLERANCE,
                                **kwargs):
    """
    Factory returning a pipeline to preprocess raster data for training.

    Applies operations for selecting, resampling, tiling and shuffling raw RGB
    dask arrays extracted from rasters into a stack of patches ready for division
    into batches and final-stage training augmentations. An explicit batching
    is excluded to allow for doing this operation downstream together with the
    appropriate masks.

    If the labelled training examples are also images, these should be
    subjected to a subset of the same operations and shuffled by the same
    random seed. For this there exist sister-methods covering common use cases
    and with the same arguments, e.g. for binary semantic segmentation
    `get_binary_mask_training_pipeline`

    Arguments
    ---------
    window_dims: array_like
        The desired row, col size of patches used for training. Together with
        the downstream batch_size, this is usually limited by GPU memory.
        For example, (256,256) might permit O(10) images per batch.
    resample_factor: float
        see ImageResampler
    overlap_tiles: bool
        Toggles whether to use half-step overlapping patches rather than no-overlap.
        Increases dataset size by factor of ~4
    preserve_int: bool
        see ImageResampler
    seed: int
        Random seed for shuffling patches
    prepend_steps: array_like
        Any additional transformers which must be applied first. This
        facilitates e.g. the insertion of dataset-specific channel preselections.
    resample_tolerance: float
        see ImageResampler

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline encapsulating the preprocessing stages
    """
    steps = [
        ('channel_selector', ChannelSelector(channels=list(channels))),
        ('padder', ArrayPadder(*window_dims)),
        ('resampler', ImageResampler(sf=resample_factor,
                                     preserve_int=preserve_int,
                                     resample_tolerance=resample_tolerance)) 
    ]
    if overlap_tiles:
        steps.append(('tiler', OverlappingTiler(window_dims)))
    else:
        steps.append(('tiler', Tiler(window_dims)))
    if seed is not None:
        steps.append(('synchronised_shuffler', SynchronisedShuffler(seed=seed)))
    if prepend_steps:
        steps[:0] = prepend_steps
    pipeline = Pipeline(steps=steps)
    return pipeline


def get_binary_mask_training_pipeline(window_dims=(256,256),
                                      resample_factor=1.0,
                                      overlap_tiles=True,
                                      preserve_int=True,
                                      seed=42,
                                      prepend_steps=[],
                                      resample_tolerance=RESAMPLE_TOLERANCE,
                                      **kwargs):
    """
    Factory returning a pipeline to preprocess binary mask rasters for training.

    Applies operations for selecting, resampling, tiling and shuffling raw 2D
    dask arrays extracted from rasters into a stack of patches ready for division
    into batches and final-stage training augmentations. An explicit batching
    is excluded to allow for doing this operation downstream together with the
    appropriate images.

    When used in conjunction with images representing training inputs, these
    should be subjected to a subset of the same operations and shuffled by the
    same random seed. For this there exists the sister-method with the same
    arguments `get_image_training_pipeline`

    Arguments
    ---------
    window_dims: array_like
        The desired row, col size of patches used for training. Together with
        the downstream batch_size, this is usually limited by GPU memory.
        For example, (256,256) might permit O(10) images per batch.
    overlap_tiles: bool
        Toggles whether to use half-step overlapping patches rather than no-overlap.
        Increases dataset size by factor of ~4
    resample_factor: float
        see ImageResampler
    preserve_int: bool
        see ImageResampler
    seed: int
        Random seed for shuffling patches
    prepend_steps: array_like
        Any additional transformers which must be applied first. This
        facilitates e.g. the insertion of dataset-specific channel preselections.
    resample_tolerance: float
        see ImageResampler

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline encapsulating the preprocessing stages
    """
    steps = [
        ('dimension_adder', DimensionAdder()),
        ('padder', ArrayPadder(*window_dims, constant_values=0)),
        ('resampler', ImageResampler(sf=resample_factor,
                                     preserve_int=preserve_int,
                                     is_mask=True,
                                     resample_tolerance=resample_tolerance))
    ]
    if overlap_tiles:
        steps.append(('tiler', OverlappingTiler(window_dims)))
    else:
        steps.append(('tiler', Tiler(window_dims)))
    if seed is not None:
        steps.append(('synchronised_shuffler', SynchronisedShuffler(seed=seed)))
    if prepend_steps:
        steps = prepend_steps + steps
    # always reshape to row * col * channel first
    #steps = [('dimension_adder', DimensionAdder())] + steps
    pipeline = Pipeline(steps=steps)
    return pipeline


class BinariserRGB(TransformerMixin, BaseEstimator, tm.Logged):
    """
    Transformer which binarises an image/mask where channels match target_rgb

    Elements where the last axis matches the specified `target_rgb` param will
    become 1, otherwise zero.

    Works for Dask or Numpy arrays.

    Attributes
    ----------
    target_rgb : array_like
        constant vector representing the last axis of an array, typically a
        3-channel RGB value
    """
    def __init__(self, target_rgb):
        self.target_rgb = target_rgb
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        pkg = da if isinstance(X, da.Array) else np
        return pkg.all(X == self.target_rgb, axis=-1).astype(np.uint8)


def rescale_image_array(array, sf=0.5, preserve_int=True):
    """
    Applies skimage.rescale to an image-like dask array by mapping over blocks

    Parameters
    ----------
    array: array_like
        Dask array with shape (rows, columns, channels)
    sf: float
        Resampling factor: 0.5 => half spatial resolution
    preserve_int: bool
        Ensure that 8-bit integer inputs produce (rounded) 8-bit integer outputs

    Returns
    -------
    dask.array.Array
        Resampled dask image array
    """
    rescale_by_factor = partial(rescale, scale=(sf, sf, 1))
    chunk_cfg = (
        # scale the chunks in the height and width dimensions
        *tuple(tuple(int(sf * e) for e in c) for c in array.chunks[:-1]),
        # leave channels alone
        array.chunks[-1]
    )
    rescaled = array.map_blocks(rescale_by_factor,
                                dtype=np.float32,
                                chunks=chunk_cfg)
    if array.dtype == np.uint8 and preserve_int:
        log.debug("Casting resampled array back to 8-bit integers...")
        rescaled = da.rint(rescaled*255.).astype(np.uint8)
    return rescaled


class ImageResampler(TransformerMixin, BaseEstimator, tm.Logged):
    """
    Transformer for applying skimage.rescale to an image-like dask array

    Attributes
    ----------
    sf: float
        Resampling factor: 0.5 => half spatial resolution
    preserve_int: bool, optional
        Ensure that 8-bit integer inputs produce (rounded) 8-bit integer outputs
    resample_tolerance: float, optional
        If sf is within resample_tolerance of 1.0, don't resample.

    """
    def __init__(self, sf,
                 preserve_int=True,
                 resample_tolerance=RESAMPLE_TOLERANCE,
                 is_mask=False,
                 binary_mask_frac=0.1 # like all_touched, mask == 1 if
                                      # rescaled array value > binary_mask_frac
        ):
        self.sf = sf
        self.preserve_int = preserve_int
        self.resample_tolerance = resample_tolerance
        self.is_mask = is_mask
        self.binary_mask_frac = binary_mask_frac # like "all_touched", count pixels > 0.1 as 1.0 for mask

    def fit(self, X, y=None):
        if abs(self.sf - 1.0) > self.resample_tolerance:
            self.skip_ = False
            self.chunk_cfg_ = (
                # scale the chunks in the height and width dimensions
                *tuple(
                    tuple(round(self.sf * e) for e in c) for c in X.chunks[:-1]
                ),
                # leave channels alone
                X.chunks[-1]
            )
            #print(f"resampler fit with chunk cfg: {self.chunk_cfg}")
        else:
            self.skip_ = True
            self.chunk_cfg_ = X.chunks
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Input Dask array (upon which fit was called)

        Returns
        -------
        dask.array.Array
            Resampled dask image array
        """
        if self.skip_:
            return X
        rescale_by_factor = partial(rescale, scale=(self.sf, self.sf, 1))
        # NOTE: multiplying by float 1 seems to avert issues where
        # occassionally masks are zero everywhere after rescaling?
        if self.is_mask:
            X=X*1.
        rescaled = X.map_blocks(rescale_by_factor,
                                dtype=np.float32,
                                chunks=self.chunk_cfg_)
        if self.preserve_int:
            if X.dtype == np.uint8 and not self.is_mask:
                self.log.debug("Casting resampled array back to 8-bit integers...")
                #rescaled = (rescaled*255.> self.grace_factor).astype(np.uint8)
                rescaled = da.rint(rescaled*255.).astype(np.uint8)
            elif self.preserve_int and self.is_mask:
                rescaled = (rescaled > self.binary_mask_frac).astype(np.uint8)
        return rescaled


class Tiler(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer batching a (H, W, C) image array into (B, h, w, C); h, w < H, W

    Convert a 3D image array into 4D array of tiles with shape window_dims
    (for use in e.g. ML inference) in the first two dimensions (w, h if channels
    last).

    Includes uneven edges and corners which don't divide nicely into tiles by
    repeating the necessary rows and columns back from the edge to form full
    tiles.

    Invertible.

    As a consequence of using this in ML inference, parts of the edges/corners
    will be slightly oversampled by some fraction of a window, ie the same edge
    region may appear twice in two different training tiles.

    Attributes
    ----------
    window_dims: array_like
        (h, w); the dimensions of the small tiles making up the batches
    """
    def __init__(self, window_dims):
        self.window_dims = window_dims

    def fit(self, X, y=None):
        # identify the part that when cropped will nicely divide into windows
        self.cropper_ = WindowFitter(window_dims=self.window_dims).fit(X)
        # prepare transformation of cropped area to stacked windows
        self.log.debug(f"Shape going into stacker {X.shape}")
        self.stacker_ = TileStacker(self.window_dims).fit(
            X[:self.cropper_.row_max_, :self.cropper_.col_max_]
        )
        # rows and columns
        self.extra_rows_ = X[self.cropper_.row_max_:]
        self.extra_cols_ = X[:, self.cropper_.col_max_:]
        # if there are additional rows that don't fit cleanly into the window-cropped area
        if self.extra_rows_.size > 0:
            # grab the last possible set of rows forming full windows
            self.last_window_rows_ = X[-self.window_dims[0]:]
            self.last_rows_wf_ = WindowFitter(window_dims=self.window_dims)
            self.last_window_rows_ = self.last_rows_wf_.fit_transform(
                self.last_window_rows_
            )
            # figure out how to partition it into full windows
            self.r_stacker_ = TileStacker(self.window_dims).fit(self.last_window_rows_)
            # if the columns do not evenly divide into window breadths,
            # this should exclude a partial window at the (bottom) right
             # figure out indices of windows from privileged rows and columns at edge
            self.lr_max_ind_ = self.last_rows_wf_.n_windows_r_ * self.window_dims[0]
        # ditto for columns
        if self.extra_cols_.size > 0:
            # grab the last possible set of cols forming full windows
            self.last_window_cols_ = X[:, -self.window_dims[1]:]
            self.last_cols_wf_ = WindowFitter(window_dims=self.window_dims)
            self.last_window_cols_ = self.last_cols_wf_.fit_transform(
                self.last_window_cols_
            )
            # figure out how to partition it into full windows
            self.c_stacker_ = TileStacker(self.window_dims).fit(self.last_window_cols_)
            # if the rows do not evenly divide into window heights,
            # this should exclude a partial window at the bottom (right)
            # figure out indices of windows from privileged rows and columns at edge
            self.lc_max_ind_ = self.last_cols_wf_.n_windows_c_ * self.window_dims[1]
        # if neither dimension factorises into windows we need to treat the bottom
        # right corner separately
        if self.extra_rows_.size > 0 and self.extra_cols_.size > 0:
            self.br_corner_ = X[-self.window_dims[0]:, -self.window_dims[1]:]
        # we just treat this as one window on its own, with the bottom
        # right corner aligned with that of the full array
        # figure out the indices in the eventual 4D array which correspond to
        # each component
        # elements up to this index can be nicely retiled
        self.crop_max_ix_ = self.cropper_.n_windows_r_ * self.cropper_.n_windows_c_
        # figure out indices of windows from privileged rows and columns at edge
        #self.lr_max_ind_ = self.last_rows_wf_.n_windows_r_ * self.window_dims[0]
        #self.lc_max_ind_ = self.last_cols_wf_.n_windows_c_ * self.window_dims[1]
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)

        Returns
        -------
        dask.array.Array
            Tiled (B, h, w, C) Dask array
        """
        # anticipate output arrays
        X_out_components = []
        # first extract the nicely dividing part
        X_crop = self.cropper_.transform(X)
        # tile it
        self.X_crop_tiled = self.stacker_.transform(X_crop)
        # tile the extra rows
        X_out_components.append(X_crop_tiled)
        if self.extra_rows_.size > 0:
            X_out_components.append(
                self.r_stacker_.transform(self.last_window_rows_)
            )
        if self.extra_cols_.size > 0:
            X_out_components.append(
                self.c_stacker_.transform(self.last_window_cols_)
            )
        if self.extra_rows_.size > 0 and self.extra_cols_.size > 0:
            X_out_components.append(
                self.br_corner_.reshape((1, *self.br_corner_.shape))
            )
        if isinstance(X, da.Array):
            lib = da
        elif isinstance(X, np.ndarray):
            lib = np
        else:
            raise ValueError("array data type not understood")
        return lib.concatenate(X_out_components)

    def inverse_transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Tiled (B, h, w, C) Dask array

        Returns
        -------
        dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)
        """
        # determine whether to use numpy or dask
        if isinstance(X, da.Array):
            lib = da
        elif isinstance(X, np.ndarray):
            lib = np
        else:
            raise ValueError(
                f"array type {type(X)} not understood. expected np or dask array"
            )
        X_crop_tiled, X_rest_tiled = X[:self.crop_max_ix_], X[self.crop_max_ix_:]
        X_crop = self.stacker_.inverse_transform(X_crop_tiled)
        # stop here if array perfectly factorised into windows
        if self.extra_rows_.size == 0 and self.extra_cols_.size == 0:
            return X_crop
        # reconstruct the partial window of rows at the bottom edge
        if self.extra_rows_.size > 0:
            ix_lr = self.last_rows_wf_.n_windows_c_
            X_last_window_rows = self.r_stacker_.inverse_transform(
                X_rest_tiled[:ix_lr]
            )
            # get the non-duplicated part of the window to stitch back on
            X_last_window_rows = X_last_window_rows[
                -self.extra_rows_.shape[0]:
            ]
            # if there are no extra columns, concatenate and return here
            if self.extra_cols_.size == 0:
                return lib.concatenate([X_rest, X_last_window_rows], axis=0)
            X_rest_tiled = X_rest_tiled[ix_lr:]
        # reconstruct the partial window of columns at the rightmost edge
        if self.extra_cols_.size > 0:
            ix_lc = self.last_cols_wf_.n_windows_r_
            X_last_window_cols = self.c_stacker_.inverse_transform(
                X_rest_tiled[:ix_lc]
            )
            # get the non-duplicated part of the window to stitch back on
            X_last_window_cols = X_last_window_cols[
                :, -self.extra_cols_.shape[1]:
            ]
            # if there are no extra rows, concatenate and return here
            if self.extra_rows_.size == 0:
                return lib.concatenate([X_rest, X_last_window_cols], axis=1)
        # otherwise we need to stick the corner to a row/col first before conc.
        br_corner = X_rest_tiled[
            -1, -self.extra_rows_.shape[0]:, -self.extra_cols_.shape[1]:
        ]
        # workaround - block doesn't seem to work here? or im just a dafty
        c1=da.concatenate([X_crop, X_last_window_rows])
        c2=da.concatenate([X_last_window_cols, br_corner])
        return lib.concatenate([c1, c2], axis=1)


# version with assertions
class Tiler(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer batching a (H, W, C) image array into (B, h, w, C); h, w < H, W

    Convert a 3D image array into 4D array of tiles with shape window_dims
    (for use in e.g. ML inference) in the first two dimensions (w, h if channels
    last).

    repeating the necessary rows and columns back from the edge to form full
    tiles.

    Invertible.

    As a consequence of using this in ML inference, parts of the edges/corners
    will be slightly oversampled by some fraction of a window, ie the same edge
    region may appear twice in two different training tiles.

    Parameters
    ----------
    window_dims: array_like
        (h, w); the dimensions of the small tiles making up the batches

    Attributes
    ----------
    cropper_: WindowFitter
        WindowFitter transformer to crop evenly-dividing piece of the array
    stacker_: TileStacker
        TileStacker used to divide up the part of the array that fits neatly
        into an integer numbe rof tilers
    extra_rows_: array_like
        The excess rows at the end which don't fully form a tile
    extra_cols_: array_like
        The excess columns at the end which don't fully form a tile

    Notes
    -----
    Uses 3 stackers and croppers to tile the evenly-dividing piece of the array,
    and the extra rows/columns respectively, plus a fourth object to address
    the bottom right corner.

    See fit method implementation for full details of the intermediate stages
    """
    def __init__(self, window_dims):
        self.window_dims = window_dims

    def fit(self, X, y=None):
        # identify the part that when cropped will nicely divide into windows
        self.cropper_ = WindowFitter(window_dims=self.window_dims).fit(X)
        # prepare transformation of cropped area to stacked windows
        self.log.debug(f"Shape going into stacker: {X.shape}")
        self.stacker_ = TileStacker(self.window_dims).fit(
            X[:self.cropper_.row_max_, :self.cropper_.col_max_]
        )
        # rows and columns
        self.extra_rows_ = X[self.cropper_.row_max_:]
        self.extra_cols_ = X[:, self.cropper_.col_max_:]
        row_remainder = (X.shape[0] + self.extra_rows_.shape[0]) % self.window_dims[0]
        col_remainder = (X.shape[1] + self.extra_cols_.shape[0]) % self.window_dims[1]
        #assert row_remainder == 0, (
        #    "reassembled row dimensions don't factorise: "
        #    f"({X.shape[0]} + {self.extra_rows_.shape[0]}) % {self.window_dims[0]} = {row_remainder}"
        #)
        #col_remainder = (self.cropper_.col_max_ + self.extra_cols_.shape[0]) % self.window_dims[1]
        #assert col_remainder == 0, (
        #    "reassembled col dimensions don't factorise: "
        #    f"({X.shape[1]} + {self.extra_cols_.shape[1]}) % {self.window_dims[1]} = {col_remainder}"
        #)
        # if there are additional rows that don't fit cleanly into the window-cropped area
        if self.extra_rows_.size > 0:
            # grab the last possible set of rows forming full windows
            self.last_window_rows_ = X[-self.window_dims[0]:]
            self.last_rows_wf_ = WindowFitter(window_dims=self.window_dims)
            self.last_window_rows_ = self.last_rows_wf_.fit_transform(
                self.last_window_rows_
            )
            # figure out how to partition it into full windows
            self.r_stacker_ = TileStacker(self.window_dims).fit(self.last_window_rows_)
            # if the columns do not evenly divide into window breadths,
            # this should exclude a partial window at the (bottom) right
             # figure out indices of windows from privileged rows and columns at edge
            self.lr_max_ind_ = self.last_rows_wf_.n_windows_r_ * self.window_dims[0]
        # ditto for columns
        if self.extra_cols_.size > 0:
            # grab the last possible set of cols forming full windows
            self.last_window_cols_ = X[:, -self.window_dims[1]:]
            self.last_cols_wf_ = WindowFitter(window_dims=self.window_dims)
            self.last_window_cols_ = self.last_cols_wf_.fit_transform(
                self.last_window_cols_
            )
            # figure out how to partition it into full windows
            self.c_stacker_ = TileStacker(self.window_dims).fit(self.last_window_cols_)
            # if the rows do not evenly divide into window heights,
            # this should exclude a partial window at the bottom (right)
            # figure out indices of windows from privileged rows and columns at edge
            self.lc_max_ind_ = self.last_cols_wf_.n_windows_c_ * self.window_dims[1]
        # if neither dimension factorises into windows we need to treat the bottom
        # right corner separately
        if self.extra_rows_.size > 0 and self.extra_cols_.size > 0:
            self.br_corner_ = X[-self.window_dims[0]:, -self.window_dims[1]:]
        # we just treat this as one window on its own, with the bottom
        # right corner aligned with that of the full array
        # figure out the indices in the eventual 4D array which correspond to
        # each component
        # elements up to this index can be nicely retiled
        self.crop_max_ix_ = self.cropper_.n_windows_r_ * self.cropper_.n_windows_c_
        # figure out indices of windows from privileged rows and columns at edge
        #self.lr_max_ind_ = self.last_rows_wf_.n_windows_r_ * self.window_dims[0]
        #self.lc_max_ind_ = self.last_cols_wf_.n_windows_c_ * self.window_dims[1]
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)

        Returns
        -------
        dask.array.Array
            Tiled (B, h, w, C) Dask array
        """
        # anticipate output arrays
        X_out_components = []
        # first extract the nicely dividing part
        X_crop = self.cropper_.transform(X)
        # tile it
        self.X_crop_tiled = self.stacker_.transform(X_crop)
        # tile the extra rows
        X_out_components.append(self.X_crop_tiled)
        if self.extra_rows_.size > 0:
            X_out_components.append(
                self.r_stacker_.transform(self.last_window_rows_)
            )
        if self.extra_cols_.size > 0:
            X_out_components.append(
                self.c_stacker_.transform(self.last_window_cols_)
            )
        if self.extra_rows_.size > 0 and self.extra_cols_.size > 0:
            X_out_components.append(
                self.br_corner_.reshape((1, *self.br_corner_.shape))
            )
        if isinstance(X, da.Array):
            lib = da
        elif isinstance(X, np.ndarray):
            lib = np
        else:
            raise ValueError("array data type not understood")
        return lib.concatenate(X_out_components)

    def inverse_transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Tiled (B, h, w, C) Dask array

        Returns
        -------
        dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)
        """
        # determine whether to use numpy or dask
        if isinstance(X, da.Array):
            lib = da
        elif isinstance(X, np.ndarray):
            lib = np
        else:
            raise ValueError(
                f"array type {type(X)} not understood. expected np or dask array"
            )
        X_crop_tiled, X_rest_tiled = X[:self.crop_max_ix_], X[self.crop_max_ix_:]
        X_crop = self.stacker_.inverse_transform(X_crop_tiled)
        # stop here if array perfectly factorised into windows
        if self.extra_rows_.size == 0 and self.extra_cols_.size == 0:
            return X_crop
        # reconstruct the partial window of rows at the bottom edge
        if self.extra_rows_.size > 0:
            ix_lr = self.last_rows_wf_.n_windows_c_
            X_last_window_rows = self.r_stacker_.inverse_transform(
                X_rest_tiled[:ix_lr]
            )
            # get the non-duplicated part of the window to stitch back on
            X_last_window_rows = X_last_window_rows[
                -self.extra_rows_.shape[0]:
            ]
            # if there are no extra columns, concatenate and return here
            if self.extra_cols_.size == 0:
                return lib.concatenate([X_crop, X_last_window_rows], axis=0)
            X_rest_tiled = X_rest_tiled[ix_lr:]
        # reconstruct the partial window of columns at the rightmost edge
        if self.extra_cols_.size > 0:
            ix_lc = self.last_cols_wf_.n_windows_r_
            X_last_window_cols = self.c_stacker_.inverse_transform(
                X_rest_tiled[:ix_lc]
            )
            # get the non-duplicated part of the window to stitch back on
            X_last_window_cols = X_last_window_cols[
                :, -self.extra_cols_.shape[1]:
            ]
            # if there are no extra rows, concatenate and return here
            if self.extra_rows_.size == 0:
                return lib.concatenate([X_crop, X_last_window_cols], axis=1)
        # otherwise we need to stick the corner to a row/col first before conc.
        br_corner = X_rest_tiled[
            -1, -self.extra_rows_.shape[0]:, -self.extra_cols_.shape[1]:
        ]
        # workaround - block doesn't seem to work here? or im just a dafty
        c1=da.concatenate([X_crop, X_last_window_rows])
        c2=da.concatenate([X_last_window_cols, br_corner])
        return lib.concatenate([c1, c2], axis=1)

    
class OverlappingTiler(TransformerMixin, BaseEstimator, tm.Logged):
    """
    Version of Tiler with half-window-size overlap.
    
    Constructs four basic tilers: a normal tiler, one with half-step patch size 
    offset in the height and width dimensions individually, and one with both together.
    
    Constructs arrays ~4x as large as a basic tiler as a result.
    
    Parameters
    ----------
    window_dims : 
        patch size to be constructed
    """
    def __init__(self, window_dims=(256, 256)):
        self.window_dims = window_dims
        self.half_step = tuple(int(w/2) for w in window_dims)
        self.main_tiler, self.h_tiler, self.w_tiler, self.hw_tiler = tuple(
            Tiler(self.window_dims) for _ in range(4)
        )

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)

        Returns
        -------
        dask.array.Array
            Tiled (B, h, w, C) Dask array
        """
        self.log.debug(f"Dividing array with shape {X.shape} into half-step overlapping tiles...")
        self.main_tiler.fit(X)
        self.h_tiler.fit(X[self.half_step[0]:])
        self.w_tiler.fit(X[:, self.half_step[1]:])
        self.hw_tiler.fit(X[self.half_step[0]:, self.half_step[1]:])
        return self
    
    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)

        Returns
        -------
        dask.array.Array
            Tiled (B, h, w, C) Dask array
        """
        X_main = self.main_tiler.transform(X)
        self.X_main_length_ = X_main.shape[0]
        X_h = self.h_tiler.transform(X[self.half_step[0]:])
        X_w = self.w_tiler.transform(X[:, self.half_step[1]:])
        X_hw  = self.hw_tiler.transform(X[self.half_step[0]:, self.half_step[1]:])
        self.log.debug("Assembling overlapping tiles...")
        return da.concatenate([X_main, X_h, X_w, X_hw], axis=0)
    
    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X: dask.array.Array
            Tiled (B, h, w, C) Dask array

        Returns
        -------
        dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)
        """
        return self.main_tiler.inverse_transform(X[:self.X_main_length_])    

#combined_pipeline = FeatureUnion(transformer_list=[('pipeline_1', pl1 ), ('pipeline_2', pl2 )])
class WindowFitter(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer to crop an image-like array (along the first two axes)

    The output is the subset of the input which fits into an integer multiple
    of window_dims.

    Parameters
    ----------
    window_dims: array_like, optional
        h, w of eventual window/tile size

    Attributes
    ----------
    n_windows_r_: int
        number of full window breadths along the rows
    n_windows c_: int
        number of full window breadths down the columns
    row_max_: int
        max row index
    col_max_: int
        max col index
    """
    def __init__(self, window_dims=(224,224), behaviour='crop'):
        """ defaults to adding +/- 2 pixels in each dimension """
        self.window_dims = window_dims

    def fit(self, X, y=None):
        # check args
        self.n_rows_window, self.n_cols_window = self.window_dims
        assert self.n_rows_window == self.n_cols_window, (
            "Use square window dimensions!"
        )
        # calculate number of full window-breadths in each dimension
        self.n_windows_r_, self.n_windows_c_ = [
            int(n) for n in np.array(X.shape[:2])/self.n_rows_window
        ]
        # calc indices used to crop
        self.row_max_ = self.n_windows_r_*self.n_rows_window
        self.col_max_ = self.n_windows_c_*self.n_cols_window
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array or np.ndarray
            Input (H, W, C) array (upon which fit was called)

        Returns
        -------
        dask.array.Array or np.ndarray
            Cropped (H', W', C) array; H' <= H, W' <= W
        """
        self.log.debug("Transforming input data to match integer number of "
                  "windows...")
        # get the part of the array which divides evenly into windows
        X_ = X[:self.row_max_, :self.col_max_]
        return X_


    

class TileStacker(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer batching a (H, W, C) image array into (B, h, w, C); h, w < H, W

    Convert a 3D image array into 4D array of tiles with shape window_dims
    (for use in e.g. ML inference) in the first two dimensions (w, h if channels
    last).

    This method works only where the input array neatly divides into tiles.
    For a general version dealing with cases which don't nearly divide see Tiler.

    Invertible.

    See:
    https://stackoverflow.com/questions/42297115/
    numpy-split-cube-into-cubes/42298440#42298440

    Parameters
    ----------
    hypertile_shape: array_like
        (h, w); the dimensions of the small tiles making up the batches

    Attributes
    ----------
    X_hypertile_shape_: array_like
        The full shape in 3D of each tile/patch
    X_old_shape_: array_like
        Copy of the original shape of the array
    X_repeats_: array_like
        Array with the ratios of the size to the tile size in each dimension
    X_order_: array_like
        The order of the higher-dimensional X_tmpshape axis to transpose by
        in the final step of the tiling operation
    X_tmpshape_: array_like
        An intermediary higher-dimensional shape to facilitate the tiling
        
    To Do
    -----
    Normalise the attribute names for parity with the other related tiling
    methods

    See Also
    --------
    WindowFitter
    Tiler
    """
    def __init__(self, hypertile_shape:tuple=(224,224)):
        self.hypertile_shape = hypertile_shape

    def fit(self, X, y=None):
        # e.g. if given an image 9000 * 16000 * 3 and provide hypertile size (200, 200)
        # this will be (200, 200, 3). likewise if mask is 9000 * 16000 * 2 => (200, 200, 2)
        self.X_hypertile_shape_ = np.array(list(self.hypertile_shape) + [X.shape[-1]])
        self.X_old_shape_ = np.array(X.shape)
        self.X_repeats_ = (self.X_old_shape_ / self.X_hypertile_shape_).astype(int)
        self.X_tmpshape_ = np.stack((self.X_repeats_, self.X_hypertile_shape_), axis=1)
        self.X_tmpshape_ = self.X_tmpshape_.ravel()
        self.X_order_ = np.arange(len(self.X_tmpshape_))
        self.X_order_ = np.concatenate([self.X_order_[::2], self.X_order_[1::2]])
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array or numpy.ndarray
            Input (H, W, C) array (upon which fit was called)

        Returns
        -------
        dask.array.Array or numpy.ndarray
            Tiled (B, h, w, C) array
        """
        self.log.debug("Reshaping array into tiles...")
        #log.debug(self.X_tmpshape_, self.X_order_, self.X_hypertile_shape_)
        # adapt to channel dimension of input
        X_tmpshp = (*self.X_tmpshape_[:-1], X.shape[-1])
        X_ht_shp = (*self.X_hypertile_shape_[:-1], X.shape[-1])
        # new_shape must divide old_shape evenly or else ValueError will be raised
        #log.debug(X_tmpshp, X_ht_shp)
        try:
            X_ = X.reshape(X_tmpshp).transpose(*self.X_order_).reshape(
                -1, *X_ht_shp
            )
        except Exception as e:
            self.log.error(e)
            self.log.debug(f"tmp: {self.X_tmpshape_}, ord: {self.X_order_}, "
                           f"x_ht_shp: {self.X_hypertile_shape_}")
            raise
        return X_

    def inverse_transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Tiled (B, h, w, C) array

        Returns
        -------
        dask.array.Array
            Input (H, W, C) array (upon which fit was called)
        """
        self.log.debug("Reshaping array into tiles...")
        X_N_, X_new_shape_ = X.shape[0], X.shape[1:]
        # update channel dim
        old_shp = np.array((*self.X_old_shape_[:-1], X.shape[-1]))
        #X_repeats_ = (self.X_old_shape_ / X_new_shape_).astype(int)
        X_repeats_ = (old_shp / X_new_shape_).astype(int)
        X_tmpshape_ = np.concatenate([X_repeats_, X_new_shape_])
        X_order_ = np.arange(len(X_tmpshape_)).reshape(2, -1).ravel(order='F')
        # adapt to channel dimension of input
        #X_tmpshape_[-1] = X.shape[-1]
        #X_old_shape_ = (*self.X_old_shape_[:-1], X.shape[-1])
        # transform
        #log.debug(X_tmpshape_, X_old_shape_)
        X_ = X.reshape(X_tmpshape_).transpose(*X_order_).reshape(old_shp)
        return X_


class DimensionAdder(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Simple Transformer to add a channel dimension to an array if it's 2D

    For example, array with shape (H, W) -> (H, W, 1)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if len(X.shape) == 2:
            self.log.debug("Adding a third (channel) dimension to array...")
            return X.reshape(list(X.shape) + [1])
        elif len(X.shape) == 3:
            self.log.debug("Leaving array shape alone...")
            return X
        else:
            raise ValueError("X should have two (row/col) or three dimensions (+channel)")


class SimpleInputScaler(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Simple Transformer wrapper for scaling an array

    Parameters
    ----------
    sf: float
        scale factor, typically 1/255. to map 8-bit RGBs -> [0, 1]
    """
    def __init__(self, sf=1/255.):
        self.sf = sf
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X * self.sf


class Float32er(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Simple Transformer wrapper for casting an array to 32-bit float
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.astype(np.float32)


class SynchronisedShuffler(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer to (un)shuffle array along axis 0 with a seeded permutation

    Works with Dask or Numpy arrays.

    Invertible.

    Parameters
    ----------
    seed: int
        random seed value

    Attributes
    ----------
    random_index_permutation_: array_like
        the permutation of indices specified by the random seed
    inverse_permutation_: array_like
        the inverse permutation of indices to unshuffle the array
    """
    def __init__(self, seed:int=42):
        self.seed = seed

    def fit(self, X, y=None):
        self.random_index_permutation_ = (
            np.random.RandomState(seed=self.seed).permutation(X.shape[0])
        )
        self.inverse_permutation_ = np.argsort(self.random_index_permutation_)
        return self

    def get_arr_perm(self, arr, perm):
        if isinstance(arr, np.ndarray):
            return arr[self.random_index_permutation_]
        elif isinstance(arr, da.Array):
            return da.slicing.shuffle_slice(arr, self.random_index_permutation_)

    def transform(self, X, y=None):
        X_ = self.get_arr_perm(X, self.random_index_permutation_)
        return X_

    def inverse_transform(self, X, y=None):
        X_ = self.get_arr_perm(X, self.inverse_permutation_)
        return X_


class ChunkBatchAligner(TransformerMixin, BaseEstimator, tm.Logged):
    """
    Transformer to rechunk an array along axis 0 so 1 chunk = M * batch_size

    Parameters
    ----------
    batch_size: int
        the batch size
    ideal_chunksize: int
        a precalculated "ideal" chunk size based on RAM considerations, for
        example this might be 60 with a batch_size of 7, which will result in
        9 batches * 7 samples = 63 samples per chunk, the closest integer

    Attributes
    ----------
    chk_cfg_: array_like
        the chunk shape of the reshaped array
    batches_per_chunk_: int
        the closest integer number of batches which will fit in a chunk of
        ideal_chunksize
    """
    def __init__(self, batch_size=None, ideal_chunksize=None):
        self.batch_size=batch_size
        self.ideal_chunksize = ideal_chunksize
    def fit(self, X, y=None):
        assert self.batch_size is not None
        # align specified ideal chunksize if provided, otherwise use current chunksize
        chunksize = X.chunksize[0] if self.ideal_chunksize is None else self.ideal_chunksize
        #self.divisible = True if chunksize % self.batch_size == 0 else False
        self.batches_per_chunk_ = round(chunksize / self.batch_size)
        self.chk_cfg_ = (self.batches_per_chunk_*self.batch_size, -1 , -1, -1)
        return self
    def transform(self, X, y=None):
        #if not self.divisible:
        return X.rechunk(self.chk_cfg_)
        #return X


class Rechunker(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer to adjust size of dask chunks between computations

    Parameters
    ----------
    chunks: array_like or str or int, optional
        see dask.array.Array.rechunk
    threshold, optional
        see dask.array.Array.rechunk
    block_size_limit: array_like or int, optional
        see dask.array.Array.rechunk
    """
    def __init__(self, chunks='auto', threshold=None, block_size_limit=None):
        self.chunks = chunks
        self.threshold = threshold
        self.block_size_limit = block_size_limit

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if isinstance(X, da.Array):
            return X.rechunk(chunks=self.chunks,
                             threshold=self.threshold,
                             block_size_limit=self.block_size_limit)
        return X


class ChannelSelector(TransformerMixin, BaseEstimator):
    """
    Transformer to select channels / project out slices along last axis of array

    Parameters
    ----------
    channels: array_like, optional
        Indices of channels to select
    """
    def __init__(self, channels=[0,1,2]):
        self.channels = channels
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[:,:,self.channels]


class ArrayPadder(TransformerMixin, BaseEstimator):
    """
    Transformer to pad an image array up to a certain shape with constant values

    Pad an array at the end of each of the first two dimensions with constant values
    such that each of these is at least min_rows, min_cols respectively

    Parameters
    ----------
    min_rows: int
        the minimum number of acceptable rows in the output
    max_cols: int
        the minimum number of acceptable cols in the output
    constant_values: int, optional
        the values to pad with

    Attributes
    ----------
    padding_rows_: int
        number of rows which are padded. requires fit.
    padding_cols_: int
        number of cols which are padded. requires fit.
    pad_width_: array_like
        the calculated pad_width shape parameter passed to np.pad

    Notes
    -----
    If used on an array from a raster, the choice of using the end of each dim
    will not shift the origin (top left), so the geotransform is unchanged
    """
    def __init__(self, min_rows, min_cols, constant_values=1):
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.constant_values = constant_values

    def fit(self, X, y=None):
        assert len(X.shape) == 3, (
            f"shape of X should be (rows, cols, channels). got {X.shape}!"
        )
        self.padding_rows_ = max(self.min_rows - X.shape[0], 0)
        self.padding_cols_ = max(self.min_cols - X.shape[1], 0)
        self.pad_width_ =(
           (0, self.padding_rows_), # start, end (both axes)
           (0, self.padding_cols_),
           (0, 0) # do not pad channels
        )
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            lib = np
        elif isinstance(X, da.Array):
            lib = da
        else:
            raise TypeError(f"Type {type(X)} of {X} not understood in pad."
                            " Should be a numpy/dask array.")
        return lib.pad(X,
                       pad_width=self.pad_width_,
                       mode='constant',
                       constant_values=self.constant_values)



def get_partition_indices(n,
                          split_fracs:tuple=(0.7, 0.15, 0.15),
                          divisible_by=None):
    """
    Calculates indices of the boundaries of an N-fold partition of an array

    Used for splitting up e.g. train/test/validation sets

    Parameters
    ----------
    n : int
        the length of an array along an axis which will be partitioned
    split_fracs : array_like, optional
        the fraction of elements in each partition, must sum to 1. e.g. (.5, .5)
    divisible_by : int, optional
        an integer into which partitions (up to the last) must divide
        
    Returns
    -------
    list
        a list of boundary indices of the partitions
    """
    # sanity check split fractions: partitions should sum to 100%
    assert sum(split_fracs) == 1.0
    split_fracs = np.array(split_fracs)
    # add a zero to the beginning to aid forming index partitions
    if split_fracs[0] != 0.0:
        split_fracs = np.concatenate([[0.0], split_fracs])
    # get a partitioning of these according to split_fracs
    if divisible_by:
        n_chunk = len(np.arange(0, n, divisible_by))
        partitioning_ixs = np.cumsum(divisible_by*n_chunk*split_fracs)
    else:
        partitioning_ixs = np.cumsum(n*split_fracs)
    # round and convert to integers for indexing
    pixs = np.round(partitioning_ixs).astype(np.int64)
    return pixs


def split_array(arr, split_fracs:tuple=(0.7, 0.15, 0.15), axis=0):
    """
    Split an array into N partitions each with a relative fraction of total size

    Parameters
    ----------
    arr: array_like
        Target array to be split into partitions
    split_fracs: array_like
        The fraction of elements in each partition
    axis: int
        The axis along which the partitioning should occur

    Returns
    -------
    list
        sub-arrays with relative lengths of split_fracs of the total size of
        the original
    """
    inds = get_partition_indices(arr.shape[axis], split_fracs)
    slices = [slice(i1, i2) for i1, i2 in zip(inds, inds[1:])]
    return [arr[slc] for slc in slices]


# fancy pca

def channel_cov_eigh(X):
    """
    Obtain the eigendecomposition of the channel data of an image array

    Necessary step for the so-called "Fancy PCA" image augmentation technique

    Parameters
    ----------
    X: array_like
        A NumPy or Dask array with >= 2 dimensions, the last corresponding to
        the channels whose eigendecomposition will be calculated

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors) sorted in decreasing order of eval magnitude
    """
    # work with dask
    if isinstance(X, da.Array):
        lib = da
    else:
        lib = np
    # flatten to shape (N_samples*H*W , 3 (RGB)), cast to float
    X_flat = X.reshape(-1, X.shape[-1])
    if X.dtype == np.uint8:
        X_flat /= 255.
    # centre
    X_mean = X_flat.mean(axis = 0)
    X_ = X_flat - X_mean
    # calculate covariance matrix
    R = lib.cov(X_, rowvar=False)
    # compute 3x3 matrix for dask
    if hasattr(R, 'compute'):
        R = R.compute()
    # eigendecomposition
    eig_vals, eig_vecs = np.linalg.eigh(R)
    # sort eigenvectors by value and vectors correspondingly
    sort_perm = eig_vals[::-1].argsort()
    eig_vecs = eig_vecs[:, sort_perm]
    eig_vals = eig_vals[sort_perm]
    return eig_vals, eig_vecs


def sample_channel_shift(eig_vals, eig_vecs, alpha_std=0.1):
    """
    Generate a random channel shift vector to be added to each pixel of an image

    Produces a linear combination of random variable alpha (sampled from normal
    distn centred at 0 with std alpha_std) * cov eigenvalue * cov eigenvector

    Notes
    -----
    Used in fancy PCA to apply data-derived channel shifts in RGB space

    Parameters
    ----------
    eig_vals: array_like
        eigenvalues of the channel covariance matrix sorted in decreasing order
        of magnitude
    eig_vecs: array_like
        the channel-space eigenvectors to which of eig_vals applies
    alpha_std: float
        the stdev of the gaussian used to sample the eigenvalue magnitudes used
        in generating the shift


    Returns
    -------
    pert_vector: array_like
        an (n_channels,) shaped array containing the channel-perturbation
    """
    n_channels = eig_vecs.shape[0]
    # get 3x1 matrix of eigenvalues multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of alpha_std
    pert_magnitudes = np.zeros((n_channels, 1))
    # draw only once per augmentation
    alpha = np.random.normal(0, alpha_std, (n_channels,))
    # broadcast
    pert_magnitudes[:, 0] = alpha * eig_vals[:]
    # this is the vector that we're going to add to each pixel
    pert_vector = np.dot(eig_vecs.T, pert_magnitudes).T
    return pert_vector


class PCAChannelShiftSampler:
    """
    Callable which generates shift vectors with "Fancy PCA" sampling for images

    This derives the PCA decomposition of the channels from the input array, and
    acts as a callable which generates random RGB shift vectors according to
    this decomposition

    Parameters
    ----------
    X: array_like
        An image-like array with channels along the last axis. Used to calculate
        shift vectors based on the eigendecomposition of the covariance matrix.
    alpha_std: float, optional
        The stdev of the gaussian used to sample the channel shifts according to
        the eigendecompostion. wider => bigger perturbations.

    Attributes
    ----------
    eig_vals: array_like
        the channel covariance matrix's eigenvalues in desc order of magnitude
    eig_vecs: array_like
        the channel covariance matrix's eigenvectors in the same order as evals

    Returns
    -------
    array_like
        an (n_channels,) shaped random shift vector
    """
    def __init__(self, X, alpha_std=0.1, max_samples=5000):
        self.X = X
        self.max_samples = max_samples
        self.alpha_std = alpha_std
        self.calculate_eigenvectors()

    def calculate_eigenvectors(self):
        if self.max_samples:
            self.eig_vals, self.eig_vecs = channel_cov_eigh(self.X[:self.max_samples])
        else:
            self.eig_vals, self.eig_vecs = channel_cov_eigh(self.X)

    def __call__(self):
        return sample_channel_shift(eig_vals=self.eig_vals,
                                    eig_vecs=self.eig_vecs,
                                    alpha_std=self.alpha_std)

    def __repr__(self):
        return f'PCAChannelShiftSampler({self.X}, alpha_std={self.alpha_std})'


def apply_channel_shift(img, sampler:PCAChannelShiftSampler):
    """
    Generates a random channel shift vector and applies it to an image array

    Uses 'fancy PCA' sampler and perturbs input image array in channel space

    Arguments
    ---------
    img: array_like
        input array with channels along the last dimension
    sampler: PCAChannelShiftSampler
        a sampler instance

    Returns
    -------
    array_like
        input image + sampled perturbation, applied globally to all pixels
    """
    shift = sampler()
    if img.dtype == np.uint8:
        img_out = img + 255.*shift
        img_out = np.clip(img_out, 0.0, 255.0)
        return np.rint(img_out).astype(np.uint8)
    else:
        return np.clip(img + shift, 0.0, 255.0)


class FancyPCA(ImageOnlyTransform):
    """
    Augment RGB image using FancyPCA

    Parameters
    ----------
    alpha_std: float
        Sampling gaussian perturbation stdev. See sample_channel_shift.
    always_apply: bool, optional
        Whether to always apply.
        See albumentations.core.transforms_interface.ImageOnlyTransform

    Attributes
    ----------
    sampler: PCAChannelShiftSampler
        Sampler instance used to derive channel shift vectors internally

    Notes
    -----
    from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    References
    ----------
    http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
    https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """

    def __init__(self, X, alpha_std=0.3, always_apply=False, p=0.5, max_samples=1000):
        super(FancyPCA, self).__init__(always_apply=always_apply, p=p)
        self.alpha_std = alpha_std
        self.sampler = PCAChannelShiftSampler(X, self.alpha_std, max_samples=max_samples)

    @property
    def alpha_std(self):
        return self._alpha_std
        
    @alpha_std.setter
    def alpha_std(self, value):
        self._alpha_std = value
        if hasattr(self, 'sampler'):
            self.sampler.alpha_std = value
        
    def apply(self, img, alpha=0.1, **params):
        return apply_channel_shift(img, self.sampler)

    def get_params(self):
        return {"alpha_std": random.gauss(0, self.alpha_std)}

    def get_transform_init_args_names(self):
        return ("alpha_std",)

    
def balanced_oversample(image_arrays, mask_arrays, data_axis=0):
    """
    Oversamples smaller image/mask arrays to match the largest number of samples present.
    
    Calculates the largest number of samples present amongst the image arrays and mask 
    arrays given as input, then repeats elements from those with fewer samples until 
    each array has the same size and each dataset is equally represented.
    
    Parameters
    ----------
    image_arrays: list of :obj:`dask.array.Array`
        A list of image arrays, each element from a different dataset. Patches should be 
        distributed along axis 0 currently.
    mask_arrays: list of :obj:`dask.array.Array`
        A list of mask arrays, each element from a different dataset. Patches should be 
        distributed along axis 0 currently. The ground truth for image_arrays.
    data_axis: int, optional
        The axis along which the samples are distributed. Currently only 0 implemented.
        
    Returns
    -------
    tuple of list of :obj:`da.array.Array`
        The input lists of arrays, with each element now having the same size.
    """
    if data_axis != 0:
        raise NotImplementedError
    
    sizes = [a.shape[data_axis] for a in image_arrays]
    largest_size, ix_largest = max(sizes), np.argmax(sizes)
    largest_im_arr = image_arrays[ix_largest]
    largest_ma_arr = mask_arrays[ix_largest]
    
    ext_img_arrs = []
    ext_ma_arrs = []
    
    extend_by = [largest_size - a.shape[data_axis] for a in image_arrays]
    
    for ix in range(len(image_arrays)):
        im_arr, ma_arr = image_arrays[ix], mask_arrays[ix]
        n_repeats = int(extend_by[ix] / image_arrays[ix].shape[data_axis])
        # if more than twice the size, repeat smaller arrays first
        if n_repeats > 0 :
            im_arr = im_arr.repeat(n_repeats, axis=data_axis)
            ma_arr = ma_arr.repeat(n_repeats, axis=data_axis)
        # repeat a certain number of elements to fully match the sizes
        n_extra_elements = largest_size - im_arr.shape[data_axis]
        if n_extra_elements > 0:
            im_arr = da.concatenate([im_arr, im_arr[:n_extra_elements]], axis=data_axis)
            ma_arr = da.concatenate([ma_arr, ma_arr[:n_extra_elements]], axis=data_axis)
        ext_img_arrs.append(im_arr)
        ext_ma_arrs.append(ma_arr)
    return ext_img_arrs, ext_ma_arrs