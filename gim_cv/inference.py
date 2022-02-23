"""
Inference

This module provides high-level classes for creating image datasets and running
inference on them using a tensorflow.keras model (which implements fit and
accepts input of shape (batch * height * width * channels)).

Examples
--------
A very simple end to end example::

    >> # assuming you have a tf.keras model defined somewhere along with a
    >> # function which returns a pipeline which preprocesses image files
    >> # into (b,  h, w, c)
    >> from models import my_model
    >> from preprocessing import get_pipeline
    >>
    >> # say you have a bunch of tifs with the same size in this directory
    >> image_paths = [f for f in Path(./data/my_image_dataset).glob('\*.tif')]
    >> ids = InferenceDataset(image_paths, image_pipeline_factory=get_pipeline)
    >> # queue up necessary preprocessing and batching operations on files
    >> ids.prepare()
    >> # queue up GPU inference, initialise interfaces to write output files
    >> ids.schedule_inference(my_model, output_directory=Path('./output_masks'))
    >> # run inference and write output masks
    >> ids.write_mask_rasters(overwrite=False)

Section breaks are created with two blank lines. Section breaks are also
implicitly created anytime a new section starts. Section bodies *may* be
indented:

Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created by
resuming unindented text.

Attributes
----------
module_level_variable1 : int
    Module level variables may be documented in either the ``Attributes``
    section of the module docstring, or in an inline docstring immediately
    following the variable.

    Either form is acceptable, but the two should not be mixed. Choose
    one convention to document module level variables and be consistent
    with it.


.. _NumPy Documentation HOWTO:
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""
import abc
import shutil
import gc
import warnings
import os
import time #LN

import numpy as np
import dask
import dask.array as da
import tensorflow.keras

import gim_cv.config as cfg
import gim_cv.preprocessing as preprocessing

from pathlib import Path
from time import perf_counter as pc

from dask.distributed import Client, as_completed
from gim_cv.preprocessing import Tiler, get_image_inference_pipeline
from gim_cv.utils import require_attr_true
from gim_cv.dask_tools import pair_chunk_generator
from gim_cv.interfaces import get_interface
from gim_cv.interfaces.base import rescale_metadata
from gim_cv.interfaces.base import ArrayCache

import timbermafia as tm
import logging

log = logging.getLogger(__name__)


def has_empty_raster(ids:'InferenceDataset') -> bool:
    """
    Returns True if an training dataset image array is all 255s,
    i.e. an empty raster. Loads the array interfaces in order to compute this.
    """
    ids.load_input_array()
    if ids.image_reader.array.min().compute() == 255:
        log.info(f"Identified empty input raster: {ids.image_src}")
        return True
    return False        


class BaseInferenceDataset(tm.Logged, metaclass=abc.ABCMeta):
    """
    Implements common methods for single- and multi-raster 
    (Composite)InferenceDataset objects.
    
    """
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, ix):
        """ getitem accesses the tiles which the input image
            is broken into for inference
        """
        return self.X[ix]


def tar_mask_path(image_src,
                  model_str,
                  ext=cfg.mask_write_format,
                  resample_sf=None,
                  directory=cfg.output_binary_mask_raster_dir):
    """ generate a target mask file from image path, model id string, extension and target dir """
    image_filename, in_ext = Path(image_src).parts[-1].split('.')
    rs_str = '' if resample_sf is None else f'_resampled_{resample_sf}'
    return Path(directory) / Path(f"{image_filename}{rs_str}_{model_str}.{ext}")


class CompositeInferenceDataset(BaseInferenceDataset, tm.Logged):
    """ Interface to multiple InferenceDatasets (e.g. with different source rasters).
        Provides methods to apply preprocessing and inference with a given model
        on a set of input rasters independently.
        
        Parameters
        ----------
        constituents: list, optional
            A list of InferenceDatasets
        prune_fn: callable, optional
            A callable which accepts a TrainingDataset, performs some calculation
            and returns True if it's to be removed from the constituents else False.
            Used for eliminating rasters which are e.g. empty (all 255s).
    """

    composition_prepared = False

    def __init__(self, constituents=[], prune_fn=has_empty_raster):
        self.constituents = list(constituents)
        self.prune_fn = prune_fn
        
    def __add__(self, other):
        if isinstance(other, InferenceDataset):
            self.constituents.append(other)
        elif isinstance(other, CompositeInferenceDataset):
            self.constituents.extend(other)
        else:
            raise ValueError(f"Type {type(other)} not understood in "
                             f"constructing composite inference dataset")
        return self

    @property
    def tags(self):
        return set(c.tag for c in self.constituents if c.tag is not None)

    @property
    def tags_str(self):
        return '_'.join(t for t in sorted(self.tags))

    def constituent_datasets_by_tag(self, tag):
        return [ds for ds in self.constituents if ds.tag == tag]
    
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
                f"Removed {nc-len(self.constituents)} InferenceDatasets, leaving {nc}."
            )
                          
        else:
            self.log.warning("Prune called but no prune_fn attribute set. Doing nothing.")
            
    @property
    def prepared(self):
        return all(c.prepared for c in self.constituents)

    def prepare(self, validate=False):
        # remove inference datasets by pruning if set
        if self.prune_fn is not None:
            self.prune()
        # prepare constituents (apply pipelines etc)
        for c in self.constituents:
            if not c.prepared:
                # chunks get optimised after combination!
                c.prepare(validate=validate)

    def schedule_inference(self, model, **kwargs):
        for c in self.constituents:
            c.schedule_inference(model, **kwargs)

    def write_mask_rasters(self, overwrite=False, skip_failures=True):
        for c in self.constituents:
            self.currently_writing_ds = c
            try:
                c.write_mask_raster(overwrite=overwrite)
            except Exception as e:
                os.remove(c.mask_writer.raster_path)
                if not skip_failures:
                    raise

    def delete_input_rasters(self):
        for c in self.constituents:
            c.delete_input_raster()

    def cache_mask_array_tiles(self, overwrite=False):
        for c in self.constituents:
            c.cache_mask_array_tiles(overwrite=overwrite)


class InferenceDataset(BaseInferenceDataset, tm.Logged):
    """
    Facilitates running segmentation inference with a model on an input raster.

    Handles preprocessing, segmenting patches, reassembling the segmented patches 
    and writing output to a new mask raster file with the georeferencing of the input.

    Parameters
    ----------
    image_src : 
        A str or Path object pointing to an image file to be segmented
    image_pipeline_factory : optional
        A function which returns a pipeline to convert image dask array into 
        preprocessed patches ready for inference
    mask_tar : optional
        Optionally specify the file to write the segmentation results to.
        If this isn't set a directory can be set instead later (see 
        :meth:`~gim_cv.inference.InferenceDataset.schedule_inference`) and 
        a reasonable output file name will be generated based on the input raster 
        and the model uuid used for inference.
    image_validation_function : optional
        Specifies a function which should accept the input image array, calculate
        some quantity and return True if valid and False if invalid.
    tag : optional
        String tag corresponding to the Dataset to which the input image belongs.
    """
    prepared = False # default state: require prepare method to be run, loading + preprocessing
    inferred = False # same idea; flag to indiciate when model inference has been run
    tile_axis = 0    # stack training images as tiles along first dimension

    def __init__(self,
                 image_src,
                 image_pipeline_factory=get_image_inference_pipeline,
                 mask_tar=None,
                 image_validation_function=None,
                 tag=None):
        # if given iterable sof associated image/shapes as arguments, construct
        # a TrainingDataset object for each pair independently then combine them
        try:
            self.image_src = Path(image_src)
        except TypeError:
            self.log.debug(
                "Attempting to interpret image/shape arguments as matching "
                "iterables"
            )
            return sum(InferenceDataset(i, s) for i, s in zip(image_src, mask_tar))
        # dispatch maskreader to shapes to produce arrays if necessary
        _im_reader_cls = get_interface(self.image_src, 'ImageReader')
        # instantiate image reader
        self.image_reader = _im_reader_cls(self.image_src, read_metadata=True)
        # if mask file target provided beforehand, instantiate writer
        self._mask_tar = mask_tar
        if mask_tar is not None:
            self.mask_tar = mask_tar
        # get references to the pipeline functions
        self.image_pipeline_factory = image_pipeline_factory
        self.image_validation_function =  image_validation_function
        self.tag = tag

    @property
    def mask_tar(self):
        return self._mask_tar

    @mask_tar.setter
    def mask_tar(self, value):
        self._mask_tar = Path(value)
        _msk_writer_cls = get_interface(self._mask_tar, 'BinaryMaskWriter')
        # instantiate mask writer
        self.mask_writer = _msk_writer_cls(self._mask_tar, metadata=self.image_reader.metadata)

    def __add__(self, other):
        """ If more than one, return a composite training dataset
        """
        if isinstance(other, InferenceDataset):
            return CompositeInferenceDataset(constituents=[self, other])
        elif isinstance(other, CompositeInferenceDataset):
            return other + self
        else:
            raise ValueError(f"Type {type(other)} not understood in "
                             f"constructing composite inference dataset")

    def load_input_array(self):
        self.log.debug(f"loading input array for {self.image_src.parts[-1]}")
        self.image_reader.load_array()
        self.log.debug(f"done loading input array for {self.image_src.parts[-1]}")

    def validate_input_array(self):
        self.log.debug(f"validating arrays")
        self.mask_writer.validate_array(self.image_validation_function)()
        self.log.debug(f"done validating arrays")

    def make_pipeline(self):
        if self.image_pipeline_factory is None:
            self.pipeline = None
        else:
            self.log.debug("Assigning pipeline...")
            self.pipeline = self.image_pipeline_factory()

    def prepare(self, validate=False):
        """
        Method to prepare data for feeding to model

        Loads image/mask arrays (with cache-reliance behaviour deferred to config
        via use_cache attr of readers), then applies some preprocessing pipeline's
        fit_transform method to the image/mask arrays

        Can skip pipeline step by explicitly setting pipeline = None

        Parameters
        ----------
        validate : 
            Boolean flag. If True, first runs image_validation_function on the input 
            raster array and raises an exception if it returns False.
        """
        self.log.debug("Obtaining arrays...")
        self.load_input_array()
        # validate is run already if arrays are cached
        if validate:
            self.validate_input_array()
        self._X = self.image_reader.array
        self.make_pipeline()
        if self.pipeline is not None:
            self._X = self.pipeline.fit_transform(self._X)
            self.log.info(f"Image pipeline for {self.image_src.parts[-1]} done!")
        self.prepared = True
        self.log.info("Input array prepared for inference!")

    @property
    def y_chunks(self):
        # copy y chunks from X
        y_chunks = np.array(self.X.chunks)
        # assumption binary mask here
        y_chunks[-1] = (1,)
        return y_chunks

    @require_attr_true('prepared')
    def schedule_inference(
        self,
        model:tensorflow.keras.Model,
        output_directory:Path=None
    ):
        """
        Creates a dask array containing segmentation masks for patches of input.

        This method maps the tf.keras model passed to it over the input patches, 
        (using a batch size of 1 which aligns with the input patch chunking)
        creating a lazy segmented patch array (assigned to y). 

        If no mask_tar output file has been specified, this will be set automatically 
        to: output_directory / image_src _ model_name _ model_checkpoint_uuid
        
        Does *not* yet compute the segmentation itself (see 
        :meth:`~gim_cv.inference.InferenceDataset.write_mask_raster`).

        Parameters
        ----------
        model : 
            A tensorflow.keras segmentation model that expects batches of patches 
            of shape: (b, h, w, c)
        output_directory : 
            A path to a directory where the output raster should be written (if the 
            path to the exact output file hasn't been specified by setting mask_tar).
            The output raster will appear in this directory with a reasonable automatically 
            generated name.
        
        """
        self.log.info(f"Scheduling inference on {self.image_src}...")
        model.stop_training = True
        # schedule inference on slices
        self._y = self.X.map_blocks(
            model.predict,
            dtype=np.float32,
            chunks=self.y_chunks
        )
        # set filename for output raster based on model used
        if self.pipeline.named_steps['resampler'].skip_:
            rs_sf = None
        else:
            rs_sf = self.pipeline.named_steps['resampler'].sf
        if self._mask_tar is None:
            path_gen_kwargs = {'resample_sf':rs_sf}
            if output_directory is not None:
                path_gen_kwargs.update(directory=output_directory)
            if model.checkpoint_uuid is not None:
                model_name = model.name + '_' + model.checkpoint_uuid + '_' + time.strftime("%Y%m%d-%H%M%S") #LN to adjust
            else:
                raise ValueError(
                    "Please assign checkpoint_uuid to model to identify "
                    "the training data and parameters used to generate results!"
                )
                # if want to be permissive, do something like
                #model_name = model.name
            self.mask_tar = tar_mask_path(
                self.image_src,
                model_name,
                 **path_gen_kwargs
            )

    def write_mask_raster(self, overwrite=False):
        """
        Computes the segmentation mask and writes it to the specified output raster.

        Runs inference on each patch of the raster, reassembles these into the full 
        raster shape and outputs to ``.tif``.

        Parameters
        ----------
        overwrite : 
            Boolean flag to control whether to skip inference if the mask_tar file
            already exists.
        """
        if not hasattr(self, '_y'):
            raise Exception("Inference not scheduled - pass a model to schedule_inference first.")
        self.log.info(f"Writing mask to file {self.mask_tar}...")
        t0 = pc()
        y_tiled = self.pipeline.named_steps['tiler'].inverse_transform(self.y)
        # slice out the part corresponding to the input array incase padding to window size
        # was necessary
        y_tiled = y_tiled[:self.image_reader.array.shape[0], :self.image_reader.array.shape[1]]
        # if rescaled, update the metadata with the rescaled version from the input image
        self.mask_writer.metadata = rescale_metadata(self.mask_writer.metadata,
                                                     new_w=y_tiled.shape[1],
                                                     new_h=y_tiled.shape[0])
        self.mask_writer.array = y_tiled
        self.mask_writer.write_raster(overwrite=overwrite)
        # ensure memory cleared
        gc.collect()
        self.log.info(f"Writing mask to file complete in {pc() - t0:.2f}s!")

    @require_attr_true('prepared')
    def get_X(self):
        return self._X

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

    def delete_input_raster(self):
        self.log.info(f"Deleting input inference raster {self.image_src}")
        os.remove(self.image_src)


def predict_on_batch_workaround(self, batch):
    """ WORKAROUND FOR MEMORY LEAK
        https://github.com/tensorflow/tensorflow/issues/33009
        https://github.com/keras-team/keras/issues/13118
    """
    return self.predict_on_batch(batch)#.numpy()
    # custom batched prediction loop to avoid memory leak issues for now in the model.predict call
    #y_pred_probs = np.empty([len(X_test), VOCAB_SIZE], dtype=np.float32)  # pre-allocate required memory for array for efficiency

    #BATCH_INDICES = np.arange(start=0, stop=len(X_test), step=BATCH_SIZE)  # row indices of batches
    #BATCH_INDICES = np.append(BATCH_INDICES, len(X_test))  # add final batch_end row

    #for index in np.arange(len(BATCH_INDICES) - 1):
    #    batch_start = BATCH_INDICES[index]  # first row of the batch
    #    batch_end = BATCH_INDICES[index + 1]  # last row of the batch
    #    y_pred_probs[batch_start:batch_end] = model.predict_on_batch(X_test[batch_start:batch_end])
