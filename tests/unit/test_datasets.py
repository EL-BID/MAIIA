import asyncio
import subprocess
import shlex
import pytest
import time
import timbermafia as tm

import gim_cv.config as cfg
import gim_cv.datasets as datasets

from pathlib import Path

import logging
log = logging.getLogger(__name__)

#from dataset_fixtures import ds_tif, ds_jp2_shp

# set fixtures to define a fake dataset (one training, one inference?)
def test_fixture_files_present(ds_jp2_shp, ds_tif):
    """Check fixture test files present"""
    assert ds_jp2_shp.image_paths
    assert ds_jp2_shp.mask_paths
    assert ds_tif.image_paths
    assert ds_tif.mask_paths


def test_list_datasets():
    """Check list_datasets catches some datasets defined in datasets.py"""
    assert datasets.list_datasets()


def test_get_dataset():
    assert datasets.get_dataset('inria')


@pytest.mark.skip
def test_get_image_inference_pipeline_by_tag():
    datasets.get_dataset('inria')


@pytest.mark.skip
def test_get_image_training_pipeline_by_tag():
    pass

@pytest.mark.skip
def test_get_binary_mask_training_pipeline_by_tag():
    pass

@pytest.mark.skip
def test_sorted_files_matching():
    pass

@pytest.mark.skip
def test_training_loader_factory():
    pass

@pytest.mark.skip
def test_inference_loader_factory():
    pass


def test_image_files_exist(ds_jp2_shp, ds_tif):
    assert ds_jp2_shp.image_files_exist
    assert ds_tif.image_files_exist


def test_all_files_exist(ds_jp2_shp, ds_tif):
    assert ds_jp2_shp.all_files_exist
    assert ds_tif.all_files_exist


def test_download_image_files(ds_download):
    asyncio.run(
        ds_download.download_image_files(
            save_dir='/tmp',
        )
    )
    assert ds_download.image_paths

def test_download_mask_files(ds_download):
    asyncio.run(
        ds_download.download_mask_files(
            save_dir='/tmp'
        )
    )
    assert ds_download.mask_paths

def test_download_all_files(ds_download):
    asyncio.run(
        ds_download.download_all_files(
            img_save_dir='/tmp',
            mask_save_dir='/tmp'
        )
    )
    assert ds_download.image_paths and ds_download.mask_paths


def test_delete_image_files():
    fake_images = ['/tmp/img_test_1_.tif', '/tmp/img_test_2_.tif']
    for f in fake_images:
        subprocess.Popen(shlex.split(f'touch {f}'))
        time.sleep(0.01)
        assert Path(f).exists()
    ds = datasets.Dataset(tag='_',
                          spatial_resolution=1.,
                          image_paths=fake_images)
    # should fail without forcing if no download function specified
    with pytest.raises(Exception) as exc_info:
        ds.delete_image_files()
    assert "download function" in str(exc_info)
    ds.delete_image_files(force=True)
    assert all (not p.exists() for p in ds.image_paths)


def test_delete_mask_files():
    fake_masks = ['/tmp/mask_test_1_.tif', '/tmp/mask_test_2_.tif']
    for f in fake_masks:
        subprocess.Popen(shlex.split(f'touch {f}'))
        time.sleep(0.01)
        assert Path(f).exists()
    ds = datasets.Dataset(tag='_',
                          spatial_resolution=1.,
                          mask_paths=fake_masks)
    # should fail without forcing if no download function specified
    with pytest.raises(Exception) as exc_info:
        ds.delete_mask_files()
    assert "download function" in str(exc_info)
    ds.delete_mask_files(force=True)
    assert all (not p.exists() for p in ds.mask_paths)


def test_get_flanders_dataset_table():
    df = datasets.get_flanders_dataset_table()
    assert df.size > 0


@pytest.mark.skip
def prepare_flanders_dataset_table():
    """ TODO: make this more isolated """
    df = datasets.get_flanders_dataset_table()
    assert 'dataset_id' in df.columns


def test_get_flanders_datasets():
    vl_ds = datasets.get_flanders_datasets()
    assert vl_ds
    assert vl_ds[0].image_download_fn is not None


def test_get_wallonia_datasets():
    wl_ds = datasets.get_wallonia_datasets()
    assert wl_ds


@pytest.mark.skip
#if(sys.version_info < (3,3),reason="requires python3.3")
def test_wallonia_datasets_exist():
    for ds in datasets.get_wallonia_datasets():
        assert ds.image_paths
        assert ds.all_files_exist


class TestTrainingDataLoader:
    @pytest.mark.skip
    def test_load(self):
        pass
    @pytest.mark.skip
    def test_load_training_data(self):
        pass

class TestInferenceDataLoader:
    @pytest.mark.skip
    def test_load(self):
        pass
    @pytest.mark.skip
    def test_load_inference_data(self):
        pass


class TestDataset:

    def test_add_datasets_fail(self, ds_jp2_shp, ds_tif):
        with pytest.raises(ValueError) as exc_info:
            _ds = ds_jp2_shp + ds_tif
        assert "spatial resolutions" in str(exc_info.value)

    def test_add_datasets(self, ds_jp2_shp, ds_tif):
        """Adding datasets of the same spatial res should make another dataset
           with the sum of their paths and masks
        """
        _sr = ds_jp2_shp.spatial_resolution
        ds_jp2_shp.spatial_resolution = ds_tif.spatial_resolution
        ds_sum = ds_jp2_shp + ds_tif
        assert isinstance(ds_sum, datasets.Dataset)
        ds_jp2_shp.spatial_resolution = _sr
        assert (len(ds_sum.image_paths) ==
            len(ds_jp2_shp.image_paths) + len(ds_tif.image_paths)
        )

    def test_files_exist(self, ds_jp2_shp, ds_tif):
        for d in (ds_jp2_shp, ds_tif):
            assert d.all_files_exist

    def test_download_image_files_fails(self, ds_jp2_shp, ds_tif):
        """Existing local files"""
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(ds_jp2_shp.download_image_files())
            asyncio.run(ds_tif.download_image_files())

    def test_download_files(self, ds_download):
        """Download files to local disks"""
        assert isinstance(ds_download, datasets.Dataset)
        asyncio.run(
            ds_download.download_all_files(img_save_dir='/tmp',
                                           mask_save_dir='/tmp')
        )
        assert ds_download.image_paths and ds_download.mask_paths
