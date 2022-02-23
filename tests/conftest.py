import pytest
import subprocess
import shlex
import logging
import os

import gim_cv.datasets as datasets
import gim_cv.config as cfg

from functools import partial


log = logging.getLogger(__name__)



# dataset fixtures

@pytest.fixture
def ds_tif():
    """A sample dataset with a TIF format RGB raster and ground truth mask"""
    return datasets.Dataset(tag='test_ds_tif',
                            spatial_resolution=0.3,
                            image_paths=[cfg.test_tif_raster],
                            mask_paths=[cfg.test_tif_mask])


@pytest.fixture
def ds_jp2_shp():
    """A sample dataset with a JPEG2000 format RGB raster and ground truth
       mask derived from vector data in a shapefile
    """
    return datasets.Dataset(tag='test_ds_jp2_shp',
                            spatial_resolution=0.25,
                            image_paths=[cfg.test_jp2_raster],
                            mask_paths=[cfg.test_shp_mask])

@pytest.fixture
def ds_download():
    """A fake dataset that has an asynchronous download function implemented
    """
    async def dl_imgs(save_dir, **kwargs):
        """ fake download function """
        files = ['test_img1.tif', 'test_img.tif']
        for n in files:
            subprocess.Popen(shlex.split(f'touch {save_dir}/{n}'))
        return [f'{save_dir}/{f}' for f in files]
    async def dl_masks(save_dir, **kwargs):
        """ fake download function """
        files = ['test_mask1.tif', 'test_mask2.tif']
        for n in files:
            subprocess.Popen(shlex.split(f'touch {save_dir}/{n}'))
        return [f'{save_dir}/{f}' for f in files]

    ds = datasets.Dataset(tag='_test_download_ds',
                          spatial_resolution=1.,
                          image_download_fn=dl_imgs,
                          mask_download_fn=dl_masks)
    yield ds
    for f in ds.image_paths + ds.mask_paths:
        os.remove(f)
