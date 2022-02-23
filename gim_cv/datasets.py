""" datasets.py

    Provides functionality for constructing ML-ready datasets from raw files

    Defines collections of input data for training or inference, locally (by
    setting file paths directly) or remotely (by specifying methods to download
    files given their URLs)

    Some pre-defined remote datasets are defined here - for example, the Flemish
    orthophotos.

    To do
    -----
    The APIs for datasets and dataloaders and dataloader factories and suchlike
    are a bit clunky atm but I haven't had time to revisit them and structure
    them more intuitively. Feel free to open a pull request!

    As more datasets are added this file will get quite monolithic. Perhaps
    split individual datasets into submodules. For this may need to slightly
    redesign the tag descriptor or introspect the module namespace for Datasets.
"""
import os
import re
import random
import pickle
import shutil
import warnings
import importlib
import hashlib
import itertools
import operator
import gc
import abc

import sqlalchemy as db
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gim_cv.config as cfg
import gim_cv.preprocessing as preprocessing

from operator import attrgetter
from pathlib import Path, PosixPath
from typing import Union, Any
from functools import partial, reduce

from cached_property import cached_property
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.url import URL

from gim_cv.preprocessing import (get_image_training_pipeline,
                                        get_binary_mask_training_pipeline,
                                        get_image_inference_pipeline,
                                        BinariserRGB)
from gim_cv.inference import InferenceDataset, CompositeInferenceDataset
from gim_cv.orchestration import download_extract_translate
from gim_cv.training import TrainingDataset, CompositeTrainingDataset
from gim_cv.utils import RegisteredAttribute, free_from_white_pixels, require_attr_true

import logging
import timbermafia as tm

log = logging.getLogger(__name__)

# global dataset registry
DATASETS = {}
BINARY_MASK_PIPELINE_PREP_STEPS = {}
IMAGE_VALIDATION_FUNCTIONS = {}
MASK_VALIDATION_FUNCTIONS = {}
# create shortcut names for long datasets
DATASET_ALIASES = {}


def list_datasets(registry=DATASETS, skip_missing_files=False):
    """
    Returns a list of the tags for all datasets currently defined in this file.

    Parameters
    ----------
    registry: dict, optional
        The registry dictionary tracking the tags of datasets added
    skip_missing_files: bool, optional
        Excludes datasets for which there are no files present locally
        and no download method assigned

    Returns
    -------
    list of str:
        The tags for every dataset included

    See Also
    --------
    get_dataset
    """
    all_tags = list(d for d in registry.keys())
    if not skip_missing_files:
        return all_tags
    tags_ = []
    for t in all_tags:
        ds = get_dataset(t)
        files_present = ds.image_paths != [] and ds.image_files_exist
        if files_present or ds.file_download_available:
            tags_.append(t)
    return tags_


def get_dataset(tag, registry=DATASETS):
    """
    Returns the dataset associated with a specific string tag.

    Parameters
    ----------
    tag: str
        An existing dataset tag, for example, "inria_austin"
    registry: dict, optional
        The registry dictionary tracking the tags of datasets added

    Returns
    -------
    Dataset:
        The dataset with the specified tag

    Raises
    ------
    KeyError:
        If no dataset exists with this tag
    """
    try:
        return registry[tag]
    except KeyError as k:
        raise AttributeError(f"no dataset with tag {tag} found!") from k


def get_image_inference_pipeline_by_tag(tag, **kwargs):
    """
    Returns dataset-appropriate pipeline for preprocessing images for inference

    Parameters
    ----------
    tag: str
        The tag of the dataset for which to get the inference pipeline

    Returns
    -------
    sklearn.pipeline.Pipeline:
        A pipeline for preprocessing this dataset's raster data

    Notes
    -----
    Written in this form to allow overriding default pipelines for some datasets
    """
    return partial(get_image_inference_pipeline, **kwargs)


def get_image_training_pipeline_by_tag(tag, **kwargs):
    """
    Returns a dataset-appropriate pipeline for preprocessing images for training

    Parameters
    ----------
    tag: str
        The tag of the dataset for which to get the training pipeline

    Returns
    -------
    sklearn.pipeline.Pipeline:
        A pipeline for preprocessing this dataset's raster data

    Notes
    -----
    Written in this form to allow overriding default pipelines for some datasets
    """
    return partial(get_image_training_pipeline, **kwargs)


def get_binary_mask_training_pipeline_by_tag(tag, **kwargs):
    """
    Returns a dataset-appropriate pipeline for preprocessing masks for inference

    Parameters
    ----------
    tag: str
        The tag of the dataset for which to get the inference pipeline

    Returns
    -------
    sklearn.pipeline.Pipeline:
        A pipeline for preprocessing this dataset's raster data

    Notes
    -----
    Written in this form to allow overriding default pipelines for some datasets
    """
    return partial(get_binary_mask_training_pipeline,
                   prepend_steps=BINARY_MASK_PIPELINE_PREP_STEPS.get(tag, []),
                   **kwargs)


def sorted_files_matching(pattern, directory):
    """
    Return a sorted list of the files in a given directory which match a regex

    Parameters
    ----------
    pattern: str
        A regex pattern for the filenames (not including parent dirs)
    directory: str or :obj:`pathlib.Path`
        The directory in which to search

    Returns
    -------
    list of :obj:`pathlib.Path` objects corresponding to the matching files
    """
    return sorted(
        [f for f in directory.glob('*') if re.match(pattern, str(f.parts[-1]))]
    )


class Dataset(tm.Logged):
    """
    Class which groups together raster and/or shapefiles into a dataset

    Datasets are identifiable via their tag attribute, and must have an
    assigned spatial resolution (to give an understanding of compatibility
    and where one needs to rescale when using more than one together)

    Datasets can be combined with the addition operator, merging their
    files into a single list if their spatial resolutions are ~the same

    Can be provided an optional image_download_fn, mask_download_fn, and/or
    all_download_fn to retrieve the files from the internet. This should be
    asynchronous to enable it to work in the background.

    all_download_fn incase it's only possible to download both sets at once
    (say, a zip archive)


    Parameters
    ----------
    tag: str
        A string identifier for this dataset
    spatial_resolution: float
        The spatial resolution of the rasters in this dataset (in a given
        Dataset, these should all be the same). Used to guide resampling
        operations downstream.
    image_paths: list of (str or :obj:`pathlib.Path`), optional
        A list of the constituent raster files.
    mask_paths: list of (str or :obj:`pathlib.Path`), optional
        A list of the constituent mask files if appropriate, in one-to-one
        correspondence with the images in image_paths. These can also be
        shapefiles, which are rasterised and converted into masks by a
        preprocessing operation downstream.
    image_download_fn: callable, optional
        A function which downloads raster files and returns a list of the paths
        of the files downloaded
    mask_download_fn: callable, optional
        A function which downloads mask files and returns a list of the paths of
        the files downloaded
    all_download_fn_fn: callable, optional
        A function which downloads both rasters and masks and returns a 2-tuple
        of the lists of paths to each respectively
    """
    tag = RegisteredAttribute('tag', registry=DATASETS)

    def __init__(self,
                 tag,
                 spatial_resolution,
                 image_paths=[],
                 mask_paths=[],
                 image_download_fn=None,
                 mask_download_fn=None,
                 all_download_fn=None,
                 eps=1e-5,
                 register_tag=True):
        self.spatial_resolution = spatial_resolution
        # option to skip registration of tag to avoid clutter
        if register_tag:
            self.tag = tag
        else:
            self.__dict__['tag'] = tag
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.eps = eps # resoluton tolerance
        # asynchronous, return a list of paths to downloaded files
        self.image_download_fn = image_download_fn
        self.mask_download_fn = mask_download_fn
        self.all_download_fn = all_download_fn

    def __add__(self, *others):
        img_pths = []
        msk_pths = []
        for o in others:
            if not abs(self.spatial_resolution - o.spatial_resolution) < self.eps:
                raise ValueError("spatial resolutions have to be the same!")
            img_pths.extend(o.image_paths)
            msk_pths.extend(o.mask_paths)
        otag = '_'.join(o.tag for o in others)
        tag = self.tag + '_' + otag
        # skip tag registration to avoid clutter from adding many datasets
        return Dataset(tag=tag,
                       spatial_resolution=self.spatial_resolution,
                       image_paths = self.image_paths + img_pths,
                       mask_paths = self.mask_paths + msk_pths,
                       register_tag=False)

    def __repr__(self):
        return f"{self.__class__.__name__}(tag={self.tag}, spatial_resolution={self.spatial_resolution})"

    @property
    def image_files_exist(self):
        return True if (
            self.image_paths and all(Path(p).exists() for p in self.image_paths)
        ) else False

    @property
    def all_files_exist(self):
        if ((self.mask_paths and all(Path(p).exists() for p in self.mask_paths)) or (not self.mask_paths) 
            and self.image_files_exist):
            return True
        return False

    async def download_image_files(self, save_dir=None, overwrite=False, **fn_kwargs):
        if self.image_download_fn is None:
            raise ValueError("no image_download_fn assigned to dataset!")
        if save_dir is not None:
            fn_kwargs.update(save_dir=save_dir)
        self.image_paths = await self.image_download_fn(overwrite=overwrite,
                                                        **fn_kwargs)

    async def download_mask_files(self, save_dir=None, overwrite=False, **fn_kwargs):
        if self.mask_download_fn is None:
            raise ValueError("no mask_download_fn assigned to dataset!")
        if save_dir is not None:
            fn_kwargs.update(save_dir=save_dir)
        self.mask_paths = await self.mask_download_fn(overwrite=overwrite,
                                                      **fn_kwargs)

    async def download_all_files(self,
                                 img_save_dir=None,
                                 mask_save_dir=None,
                                 overwrite=False,
                                 img_kw={},
                                 msk_kw={},
                                 all_kw={}):
        if self.all_download_fn is not None:
            await self.all_download_fn(img_save_dir,
                                       mask_save_dir,
                                       overwrite=overwrite,
                                       **all_kw)
        else:
            await self.download_image_files(img_save_dir,
                                            overwrite=overwrite,
                                            **img_kw)
            await self.download_mask_files(mask_save_dir,
                                           overwrite=overwrite,
                                           **msk_kw)

    @property
    def file_download_available(self):
        if self.image_download_fn is not None or self.all_download_fn is not None:
            return True
        else:
            return False

    def delete_image_files(self, force=False):
        if self.image_download_fn is not None or force:
            self.log.info(f"Deleting local image files for dataset {self.tag}...")
            for p in self.image_paths:
                os.remove(p)
        else:
            raise Exception(f"No image download function specified for dataset {self.tag}. "
                            f"Treating local deletion as an error: run with force=True if not.")

    def delete_mask_files(self, force=False):
        if self.mask_download_fn is not None or force:
            self.log.info(f"Deleting local mask files for dataset {self.tag}...")
            for p in self.mask_paths:
                os.remove(p)
        else:
            raise Exception(f"No mask download function specified for dataset {self.tag}. "
                            f"Treating local deletion as an error: run with force=True if not.")

    @require_attr_true('all_files_exist')
    def load_training_data(
        self,
        image_pipeline_factory=None,
        mask_pipeline_factory=None,
        image_validation_function=None,
        mask_validation_function=None,
        batch_size=4,
        batch_generator_fn=None,
        train_val_test_split=(0.9, 0.1),
        **pipeline_kwargs
    ):
        """
        Creates multi-file training datasets with chosen preprocessing pipelines

        Returns a (Composite)TrainingDataset from a the image/mask files of the 
        parent Dataset.

        Parameters
        ----------
        batch_size : 
            batch size
        train_val_test_split : 
            A two- or three-element tuple with fractions of data used for training,
            validation and testing. If two elements, no testing set will be used.
        batch_generator_fn : 
            A batch generator function
        image_pipeline_factory : 
            A function which returns a preprocessing pipeline for each image array
        mask_pipeline_factory : 
            A function which returns a preprocessing pipeline for each mask array
        image_validation_function : 
            todo
        mask_validation_function : 
            todo

        Returns
        -------
        :obj:`gim_cv.training.CompositeTrainingDataset`
        """
        # use default pipeline factories for this tag if no explicit ones passed
        if image_pipeline_factory is None:
            image_pipeline_factory = (
                get_image_training_pipeline_by_tag(self.tag, **pipeline_kwargs)
            )
        if mask_pipeline_factory is None:
            mask_pipeline_factory = (
                get_binary_mask_training_pipeline_by_tag(
                    self.tag,
                    **pipeline_kwargs
                )
            )
        # use default validation functions for this tag if no explicit ones passed
        if image_validation_function is None:
            image_validation_function = IMAGE_VALIDATION_FUNCTIONS.get(self.tag)
        if mask_validation_function is None:
            mask_validation_function = MASK_VALIDATION_FUNCTIONS.get(self.tag)
        # check source files present and same number of each
        if not self.image_paths and not self.mask_paths:
            raise ValueError("Empty list of images and masks!")
        assert len(self.image_paths) == len(self.mask_paths), (
            f"Need same number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)})!"
        )
        # construct one TrainingDataset per image/mask file pair
        datasets =  [TrainingDataset(im_src, msk_src,
                                     image_pipeline_factory=image_pipeline_factory,
                                     mask_pipeline_factory=mask_pipeline_factory,
                                     train_val_test_split=train_val_test_split,
                                     batch_generator_fn=batch_generator_fn,
                                     batch_size=batch_size,
                                     image_validation_function=image_validation_function,
                                     mask_validation_function=mask_validation_function,
                                     tag=self.tag)
                     for im_src, msk_src in zip(self.image_paths, self.mask_paths)]
        # combine these into a CompositeTrainingDataset
        if len(datasets) == 1:
            datasets = CompositeTrainingDataset(constituents=datasets)
        else:
            datasets = reduce(operator.add, datasets)
        return datasets

    @require_attr_true('image_files_exist')
    def load_inference_data(
        self,
        image_pipeline_factory=None,
        image_validation_function=None,
        **pipeline_kwargs
    ):
        """
        Creates multi-file inference datasets with chosen preprocessing pipelines

        Returns a (Composite)InferenceDataset from the parent dataset's rasters.

        Parameters
        ----------
        image_pipeline_factory : optional
            A function which returns a preprocessing pipeline for each image array. Uses 
            default inference pipeline for this dataset if None.
        image_validation_function : optional
            A function which accepts an image array and returns True if valid and False
            otherwise. Defers to defaults defined per dataset (see the 
            IMAGE_VALIDATION_FUNCTIONS dict which defaults to None).
        pipeline_kwargs : optional
            Any parameters which can be passed to the pipeline factory function, for 
            example inference_window_size


        Returns
        -------
        :obj:`gim_cv.inference.CompositeInferenceDataset`
        """
        # use default pipelines and validation functions if none passed
        if image_pipeline_factory is None:
            image_pipeline_factory = get_image_inference_pipeline_by_tag(self.tag, **pipeline_kwargs)
        if image_validation_function is None:
            image_validation_function = IMAGE_VALIDATION_FUNCTIONS.get(self.tag)
        # construct one InferenceDataset per image raster
        datasets =  [
            InferenceDataset(
                im_src,
                image_pipeline_factory=image_pipeline_factory,
                image_validation_function=image_validation_function,
                tag=self.tag
            )
            for im_src in self.image_paths
        ]
        self.log.debug(f"Combine datasets: {datasets}")
        if len(datasets) == 1:
            datasets = CompositeInferenceDataset(constituents=datasets)
        elif len(datasets) > 1:
            datasets = reduce(operator.add, datasets)
        else:
            self.log.warning(f"no datasets identified in load_inference_data for {self.tag}?")
        return datasets


    def print_summary(self):
        self.log.info(f"{self} with: {len(self.image_paths)} image/mask pairs")


### ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
### define datasets
### ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

# ---- test dataset definitions

# first test dataset: belgium orthophoto JP2 raster and appropriate shapefile


ds_test_jp2_shape = Dataset(
    tag='test_jp2_shape',
    spatial_resolution=0.25,
    image_paths = [
        cfg.test_jp2_raster
    ],
    mask_paths = [
        cfg.test_shp_mask
    ]
)

# second test dataset: Medellin orthophoto and mask; both are in geotiff format
ds_test_tif = Dataset(
    tag='test_tif',
    spatial_resolution=0.4,
    image_paths = [
        cfg.test_tif_raster
    ],
    mask_paths = [
        cfg.test_tif_mask
    ]
)

# second test dataset: Medellin orthophoto and mask; both are in geotiff format
ds_train_tif = Dataset(
    tag='train_tif',
    spatial_resolution=0.4,
    image_paths = [cfg.train_data_tif_raster / Path(tif) for tif in sorted(os.listdir(cfg.train_data_tif_raster))],
    mask_paths = [cfg.train_data_tif_mask / Path(tif) for tif in sorted(os.listdir(cfg.train_data_tif_mask))]
)
# second test dataset: Medellin orthophoto and mask; both are in geotiff format
ds_predict_tif = Dataset(
    tag='infer_tif',
    spatial_resolution=0.4,
    image_paths = [cfg.infer_data_tif_path / Path(tif) for tif in sorted(os.listdir(cfg.infer_data_tif_path))]
)

#
## TRAINING DATASET DEFINITIONS
#

# --- GIM topo maps (use separately from orthophotos)

ds_topo_69 = Dataset(
    tag='belgium_topo_1969',
    spatial_resolution=1.0,
    image_paths = [
        cfg.raw_data_path /
        Path('gim_topomaps/training/1969/images/1969_22_23_modified.tif')
    ],
    mask_paths = [
        cfg.raw_data_path /
        Path('gim_topomaps/training/1969/shapes/topo_area_training.shp')
    ]
)




# ---- INRIA ortho dataset
# ---- https://project.inria.fr/aerialimagelabeling/

inria_dir = cfg.training_data_path / Path('inria/AerialImageDataset/train')
inria_images_dir = inria_dir / Path('images')
inria_mask_dir = inria_dir / Path('gt')

# ---- distinguish individual regions

tyrol_ptn = 'tyrol(.*).tif'
ds_inria_tyrol = Dataset(
    tag='inria_tyrol',
    spatial_resolution=0.3,
    image_paths = sorted_files_matching(tyrol_ptn, inria_images_dir),
    mask_paths = sorted_files_matching(tyrol_ptn, inria_mask_dir)
)

vienna_ptn = 'vienna(.*).tif'
ds_inria_vienna = Dataset(
    tag='inria_vienna',
    spatial_resolution=0.3,
    image_paths = sorted_files_matching(vienna_ptn, inria_images_dir),
    mask_paths = sorted_files_matching(vienna_ptn, inria_mask_dir)
)

austin_ptn = 'austin(.*).tif'
ds_inria_austin = Dataset(
    tag='inria_austin',
    spatial_resolution=0.3,
    image_paths = sorted_files_matching(austin_ptn, inria_images_dir),
    mask_paths = sorted_files_matching(austin_ptn, inria_mask_dir)
)


chicago_ptn = 'chicago(.*).tif'
ds_inria_chicago = Dataset(
    tag='inria_chicago',
    spatial_resolution=0.3,
    image_paths = sorted_files_matching(chicago_ptn, inria_images_dir),
    mask_paths = sorted_files_matching(chicago_ptn, inria_mask_dir)
)

kitsap_ptn = 'kitsap(.*).tif'
ds_inria_kitsap = Dataset(
    tag='inria_kitsap',
    spatial_resolution=0.3,
    image_paths = sorted_files_matching(chicago_ptn, inria_images_dir),
    mask_paths = sorted_files_matching(chicago_ptn, inria_mask_dir)
)




# ---- massach buildings dataset
# ---- https://www.cs.toronto.edu/~vmnih/data/

# register some additional preprocessing steps for this one
# just looking for the buildings, which are marked by RGB value (255, 0, 0)
BINARY_MASK_PIPELINE_PREP_STEPS['massachusetts'] = [
    ('binariser', BinariserRGB((255, 0, 0)))
]
# iirc there was maybe an issue with the tiles with some of the area missing
IMAGE_VALIDATION_FUNCTIONS['massachusetts'] = free_from_white_pixels

massach_buildings_path = cfg.training_data_path / Path('mass_buildings')
ds_massa = Dataset(
    tag='massachusetts',
    spatial_resolution=1.0,
    image_paths = sorted([d for d in massach_buildings_path.glob('train/sat/*')]),
    mask_paths = sorted([d for d in massach_buildings_path.glob('train/map/*')])
)


# -- IEEE potsdam dataset

# register additional prep steps; need to select building classes w/ rgb (,,255)
BINARY_MASK_PIPELINE_PREP_STEPS['potsdam']  = [
    ('binariser', BinariserRGB((0, 0, 255)))
]

potsdam_path = cfg.training_data_path / Path('potsdam')
ds_pots = Dataset(
    tag='potsdam',
    spatial_resolution=.05,
    image_paths = sorted([f for f in potsdam_path.glob('2_Ortho_RGB/*.tif')]),
    mask_paths = sorted([f for f in potsdam_path.glob('*label.tif')])
)


# -- IEEE vaihingen dataset
# 3 bands are NIR-R-G! so don't use this
#    Impervious surfaces (R 255, 255, 255)   Building ( 0, 0, 255)    Low vegetation (RGB: 0, 255, 255)
#    Tree (RGB: 0, 255, 0)                   Car (RGB: 255, 255, 0)   Clutter/background (RGB: 255, 0, 0)
#vaihingen_path = Path(cfg.raw_data_path / 'vaihingen')
#ds_vaih = Dataset(tag='vaihingen',
#                    spatial_resolution=.09,
#                    image_paths = sorted([f for f in vaihingen_path.glob('top/*')]),
#                    mask_paths = sorted([f for f in vaihingen_path.glob('gt/*')]))

# TODO: add Zeebrugge shapefile
# TODO: add DLR fine-grained road segmentation (in raw) and DLR skyscapes (need download link)

# =================================================================================
## INFERENCE DATASETS (no masks)
# =================================================================================


# connect to the database containing URLs and metadata for remote rasters
db_config = {
    'drivername': 'sqlite',
    'database' : str(cfg.sqlite_db_path)
}
db_url = URL(**db_config)


### FLANDERS (on demand dataset) ----------------------
def get_flanders_dataset_table(db_url=db_url,
                               table_name='vlaanderen_ortho_datasets'):
    """
    Connect to a database (normally local SQLite) containing a table with
    webscraped data used to build datasets (file URLs, acquisition years, ...)
    and return a pandas dataframe of this table to be used for downloading the
    linked files etc

    Parameters
    ----------
    db_url: str, optional
        URL to SQLite database with dataset metadata table (which is webscraped
        - see scrapers)

    table_name: str, optional
        The table name for the flanders datasets in said database

    Returns
    -------
    :obj:`pandas.DataFrame`:
        dataframe of metadata used to build datasets of remote rasters

    """
    # read table from database
    engine = db.create_engine(db_url)
    metadata = db.MetaData(engine)
    # reflect db schema to MetaData
    metadata.reflect(bind=engine)
    with engine.connect() as conn:
        # get the vlaanderen ortho url data
        ortho_datasets = db.Table(table_name,
                                  metadata,
                                  autoload=True,
                                  autoload_with=engine)
        query = db.select([ortho_datasets])
        df = pd.read_sql(query, conn)
    return df

inference_vol_path_flanders = cfg.output_path
input_raster_dir_flanders = (
    inference_vol_path_flanders / Path('input_rasters/vlaanderen')
)


def prepare_flanders_dataset_table(df,
                                   suffix_regex='(K(BL_)?)?\d{1,2}$',
                                   group_datasets_by=['region','period','scale_str'],
                                   forbid_col_regexes=[('download_url', '.*50cm.*')],
                                   eyeball_check=False):
    """
    Cleans raw metadata table and creates dataset_id column for flanders orthos

    Parameters
    ----------
    df: :obj:`pandas.DataFrame`
        A dataframe containing the SQL table with raster URLs and metadata.
        Generated by the `get_flanders_dataset_table` function.
    suffix_regex: str, optional
        A Regex expression matching an area code in the raster filename
    group_datasets_by: list of str, optional
        Columns by which to group the rasters, forming dataset tags by each
        unique combination of these
    forbid_col_regexes: list of (str, str)
        List specifying (column, pattern) pairs, such that any entries in column
        which match pattern will be ignored

    Returns
    -------
    :obj:`pandas.DataFrame`:
        A cleaned dataframe of raster metadata with a dataset_id column used to
        collect rasters into datasets to be downloaded
    """
    # get files with acceptable map codes in their names
    df = df[df['suffix'].str.match(suffix_regex)]
    # eliminate entries where columns match forbidden regexes
    for col, regex in forbid_col_regexes:
        df = df[~df[col].str.match(regex)]
    # patch 30cm images' resolution
    df.loc[df[df['download_url'].str.match('.*30cm.*')].index, 'scale'] = 0.3
    df.loc[:, 'scale_str'] = (df.scale*100).astype(np.int32).astype('str') + 'cm'
    # create ids to group together all files with the same values for
    # group_datasets_by
    df['dataset_id'] = df[group_datasets_by].agg('_'.join, axis=1)
    df = df.drop('scale_str', axis=1)
    # check
    if eyeball_check:
        print("Does this look reasonable?")
        print(df.groupby('dataset_id').count())
    return df


def get_flanders_datasets(db_url=db_url,
                          table_name='vlaanderen_ortho_datasets',
                          table_prepare_fn=prepare_flanders_dataset_table,
                          img_download_dir=input_raster_dir_flanders,
                          file_download_fn=download_extract_translate,
                          overwrite=False,
                          target_spatial_resolution=1.0,
                          return_table=False,
                          download_timeout=7200,
                          **prepare_kwargs):
    """
    Get a list of Dataset objects corresponding each Flemish ortho year

    Datasets contain flanders orthophotos and their appropriate capture time,
    spatial resolution etc. These have assigned methods to download the
    associated archives on demand, extract them to obtain the rasters, and
    optionally resample these to a target spatial resolution as a postprocessing
    step.

    The URLs and metadata used to build the datasets are defined in a SQLite
    database at db_url, which is in turn built by the scraper in `gim_cv.scrapers`.

    Parameters
    ----------
    db_url: str, optional
        URL to SQLite database with dataset metadata table (which is
        webscraped - see `gim_cv.scrapers` submodule)

    table_name: str, optional
        The table name for the flanders datasets in said database

    table_prepare_fn: callable, optional
        Preprocessing function for cleaning the raw SQL table dataframe in
        pandas and returning the df

    img_download_dir: str or :obj:`pathlib.Path`, optional
        Parent directory into which to download dataset files

    file_download_fn: callable, optional
        function with signature: (save_dir, urls, filenames, timeout,
        target_scales, overwrite) which is assigned as a method of the returned
        Datasets so that they can download all the necessary files on demand
        from the appropriate URLs. this should download and extract the raster
        files and return a list of the local paths

    overwrite: bool, optional
        Flag controlling re-download/overwrite behaviour if raster files are
        identified as already present locally

    target_spatial_resolution: float or NoneType, optional
        Sets the target spatial resolution which the downloaded rasters will be
        automatically rescaled to.

    return_table: bool, optional
        Flags whether to alternatively return a 2-tuple containing a dataframe
        of the metadata pertaining to the rasters in each dataset, in addition
        to the Dataset objects themselves.

    download_timeout: int, optional
        Downloads auto-fail after this many seconds

    Returns
    -------
    datasets: list of :obj:`Dataset`
        A list of Datasets, one per ortho year present, each with the download
        method `image_download_fn` set to pull rasters from the appropriate URLs
        If the return_table argument is True, return a 2-tuple with (this list,
        raster metadata dataframe) for inspection.

    """
    df = get_flanders_dataset_table(db_url, table_name)
    # preprocess etc, clean, remove unwanted files...
    df = table_prepare_fn(df, **prepare_kwargs)
    # make dataset objects from dataframe of urls, filenames and dataset ids
    datasets = []
    gb = df.groupby('dataset_id')
    for ds_id, ixs in gb.groups.items():
        # get the rows with the appropriate urls/file metadata
        ds_df = df.loc[ixs]
        if target_spatial_resolution is not None:
            tsr = target_spatial_resolution
        else:
            target_spatial_resolution = ds_df.iloc[0].scale
        resample_factor = ds_df.iloc[0].scale / tsr
        rs_str =  f'_resampled_{int(round(resample_factor*100))}pct' if resample_factor != 1. else ''
        # create a dataset with the appropriate tag and spatial resolution,
        # and bind the custom download function
        ds = Dataset(tag = ds_id + rs_str,
                     spatial_resolution = target_spatial_resolution)
        # assign a specific download function, with the appropriate urls and paths set from db
        ds.image_download_fn = partial(
            file_download_fn,
            save_dir = img_download_dir / Path(ds_id),
            urls = ds_df.download_url.tolist(),
            filenames = ds_df.filename.tolist(),
            timeout = download_timeout,
            target_scales = itertools.repeat(resample_factor),
            overwrite = overwrite
        )
        datasets.append(ds)
    if return_table:
        return datasets, df
    return datasets



# combine
flanders_datasets = get_flanders_datasets(
    db_url,
    file_download_fn=download_extract_translate,
    overwrite=False
)

# -- add corrupted tiles as fixed dataset
flanders_datasets.append(
    Dataset(tag='Vlaanderen_2000-2003_30cm_missing_tiles_resampled_30pct',
            spatial_resolution=1.,
            image_paths = sorted(
                [f for f in (input_raster_dir_flanders / Path('Vlaanderen_2000-2003_30cm_missing_tiles')).glob('*.tif')]
            )
    )
)


### WALLONIA ----------------------
inference_vol_path_wallonia = cfg.output_path
input_raster_dir_wallonia = (
    inference_vol_path_wallonia / Path('input_rasters/wallonia')
)
wallonia_datasets = []

# iirc one or two rasters caused issues initially so were ignored, and were
# generated again down the line as an additional "missing" dataset
dodgy_rasters_94_00 = [
    input_raster_dir_wallonia /
    Path('ORTHOS_1994-2000/TIFF_RGB/ORTHO_1994_2000__38_28_W_resampled_40pct.tif'),
]

wallonia_datasets.append(
    Dataset(tag='Wallonia_1994-2000_40cm_resampled_40pct',
            spatial_resolution=1.,
            image_paths = sorted(
                [
                    f for f in (
                        input_raster_dir_wallonia / Path('ORTHOS_1994-2000')
                    ).glob('**/*.tif')
                    if Path(f) not in dodgy_rasters_94_00
                ]
            )
    )
)

wallonia_datasets.append(
    Dataset(tag='Wallonia_2006-2007_50cm_resampled_50pct',
            spatial_resolution=1.,
            image_paths = sorted(
                [
                    f for f in (
                        input_raster_dir_wallonia / Path('Orthos_2006-2007')
                    ).glob('**/*.tif')
                ]
            )
    )
)

wallonia_datasets.append(
    Dataset(tag='Wallonia_2009-2010_25cm_resampled_25pct',
            spatial_resolution=1.,
            image_paths = sorted(
                [
                    f for f in (
                        input_raster_dir_wallonia / Path('Orthos_2009-2010')
                    ).glob('**/*.tif')
                ]
            )
    )
)

wallonia_datasets.append(
    Dataset(tag='Wallonia_2013_25cm_resampled_25pct',
            spatial_resolution=1.,
            image_paths = sorted(
                [
                    f for f in (
                        input_raster_dir_wallonia / Path('2013')
                    ).glob('**/*.tif')
                ]
            )
    )
)

wallonia_datasets.append(
    Dataset(tag='Wallonia_2015_25cm_resampled_25pct',
            spatial_resolution=1.,
            image_paths = sorted(
                [
                    f for f in (
                        input_raster_dir_wallonia / Path('Ortho2015')
                    ).glob('**/*.tif')
                ]
            )
    )
)

wallonia_datasets.append(
    Dataset(tag='Wallonia_2019_25cm_resampled_25pct',
            spatial_resolution=1.,
            image_paths = sorted(
                [
                    f for f in (
                        input_raster_dir_wallonia / Path('Orthos2019')
                    ).glob('**/*.tif')
                ]
            )
    )
)

# see above "dodgy_rasters_94_00" comment
_w94mdpath = input_raster_dir_wallonia / Path('ORTHOS_1994-2000_missing_tiles')
wallonia_datasets.append(
    Dataset(tag='Wallona_1994-2000_40cm_missing_tiles_resampled_40pct',
            spatial_resolution=1.,
            image_paths=sorted(
                [f for f in _w94mdpath.glob('*_resampled_40pct.tif')]
            )
    )
)

def get_wallonia_datasets():
    """
    Returns a list of Dataset objects pointing to the wallonia orthophotos for
    each acquisition year.

    These are expected to be present locally as there's no convenient download
    links like there are for flanders.

    Returns
    -------
    list of :obj:`Dataset`
    """
    global wallonia_datasets
    return wallonia_datasets

# Frontier Development lab open dataset

# Columbia Medellin dataset 
ds_med_19_40cm_rgb = Dataset(
    tag='col_med_19_40cm',
    spatial_resolution=.4,
    image_paths = [cfg.volumes_data_path / Path('datasets/medellin_oxford_inria/Medellin_40cm.tif')],
    mask_paths = [cfg.volumes_data_path / Path('datasets/medellin_oxford_inria/Medellin_ground_truth.tif')]
)
# Sudan ElDaein dataset 
ds_eldaien_19_40cm_rgb = Dataset(
    tag='sud_eldaien_19_40cm',
    spatial_resolution=.4,
    image_paths = [cfg.volumes_data_path / Path('datasets/eldaien_oxford_inria/ElDaien_40cm.tif')],
    mask_paths = [cfg.volumes_data_path / Path('datasets/eldaien_oxford_inria/ElDaien_40cm_ground_truth_corrected.tif')]
)
# Nigeria Makoko dataset 
ds_mak_19_50cm_rgb = Dataset(
    tag='nig_mak_19_50cm',
    spatial_resolution=.5,
    image_paths = [cfg.volumes_data_path / Path('datasets/makoko_oxford_inria/Makoko_50cm.tif')],
    mask_paths = [cfg.volumes_data_path / Path('datasets/makoko_oxford_inria/Makoko_50cm_large_ground_truth.tif')]
)
