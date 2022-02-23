import pytest
import numpy as np
import dask.array as da
from pytest_mock import MockerFixture

import gim_cv.training


@pytest.mark.skip
def test_from_first_constituent():
    pass

class TestBaseTrainingDataset():
    pass

class TestCompositeTrainingDataset():
    @pytest.mark.skip
    def test_prepare(self):
        pass
    @pytest.mark.skip
    def test_batch_gen_train(self):
        pass

class TestTrainingDataset():
    @pytest.mark.skip
    def test_prepare(self):
        pass
    @pytest.mark.skip
    def test_make_pipelines(self):
        pass
    @pytest.mark.skip
    def test_load_arrays(self):
        pass
    @pytest.mark.skip
    def test_save_prepared_arrays(self):
        pass
    @pytest.mark.skip
    def test_save_prepared_arrays(self):
        pass
    @pytest.mark.skip
    def test_batch_gen_train(self):
        pass

@pytest.mark.parametrize(
    "arr, expected", 
    [
        (da.ones((16, 16, 3), dtype=np.uint8) * 255, True), # all white/empty
        (da.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8), False) # random values
    ]
)    
def test_has_empty_raster(mocker: MockerFixture, arr : da.Array, expected:bool):
    # mock training dataset
    tds = mocker.MagicMock(**{'image_src':'test.tif'})
    tds.image_reader.array = arr
    assert gim_cv.training.has_empty_raster(tds) == expected
