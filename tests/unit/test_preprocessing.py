import pytest
import dask.array as da
import numpy as np

import gim_cv.preprocessing as pp

@pytest.fixture
def test_img_arr():
    yield da.from_array(np.random.rand(1024, 1024, 3), chunks=(128, 128, 3))

@pytest.fixture
def test_mask_arr():
    yield da.from_array(np.random.rand(1024, 1024, 3), chunks=(128, 128, 3))
    
@pytest.mark.skip
def test_get_aug_datagen():
    pass

@pytest.mark.skip
def test_get_pipelines():
    pass

@pytest.mark.skip
def test_rescale_image_array():
    pass

@pytest.mark.skip
def test_get_partition_indices():
    pass

@pytest.mark.skip
def test_split_array():
    pass


class TestBinariserRGB:
    pass


class TestImageResampler:
    pass


class TestTiler:
    pass

class TestOverlappingTiler:
    def test_fit_transform(self, test_img_arr):
        ot = pp.OverlappingTiler((128, 128))
        X_ = ot.fit_transform(test_img_arr)
        assert len(X_.shape) > len(test_img_arr.shape)
        assert X_.shape[1] == 128 and X_.shape[2] == 128
    def test_inverse_transform(self, test_img_arr):
        ot = pp.OverlappingTiler((128, 128))
        X_ = ot.fit_transform(test_img_arr)
        X_o = ot.inverse_transform(X_)
        assert (X_o == test_img_arr).all().compute()

class TestWindowFitter:
    pass


class TestTileStacker:
    pass


class TestDimensionAdder:
    pass


class TestSimpleInputScaler:
    pass


class TestFloat32er:
    pass


class TestSynchronisedShuffler:
    pass


class TestChunkBatchAligner:
    pass


class TestRechunker:
    pass
