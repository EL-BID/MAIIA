import pytest

import gim_cv.interfaces.tif.rasterio
import gim_cv.interfaces.jp2.rasterio
import gim_cv.config as cfg

# --- init
@pytest.mark.skip
def test_get_interface():
    pass

# --- base
@pytest.mark.skip
def test_cache_path_from_src_filename():
    pass

@pytest.mark.skip
def test_rescale_metadata():
    pass

class TestGeoArrayFileInterface:
    @pytest.mark.skip
    def test_validate_array(self):
        pass
    @pytest.mark.skip
    def test_load_array(self):
        pass
    @pytest.mark.skip
    def test_ensure_array_cached(self):
        pass
    @pytest.mark.skip
    def test_rescale_array_and_metadata(self):
        pass

class TestArrayCache:
    @pytest.mark.skip
    def test_save(self):
        pass
    @pytest.mark.skip
    def test_read(self):
        pass
    @pytest.mark.skip
    def test_validate(self):
        pass
    @pytest.mark.skip
    def delete(self):
        pass

# --- tif init

@pytest.fixture
def tif_raster_interface():
    yield gim_cv.interfaces.tif.rasterio.RasterInterface(
        cfg.test_tif_raster
    )

class TestImageReader:
    pass

class TestBinaryMaskReader:
    pass

class TestLabelledMaskReader:
    pass

class TestImageWriter:
    pass

class TestBinaryMaskWriter:
    pass

class TestLabelledMaskWriter:
    pass

# --- jp2 init

@pytest.fixture
def jp2_raster_interface():
    yield gim_cv.interfaces.jp2.rasterio.RasterInterface(
        cfg.test_jp2_raster
    )

class TestJP2ImageReader:    

    def test_load_array(self, jp2_raster_interface):
        jp2_raster_interface.load_array()
        assert jp2_raster_interface.array is not None
        jp2_raster_interface.array = None
        
    def test_compute_array(self, jp2_raster_interface):
        jp2_raster_interface.load_array()
        arr = jp2_raster_interface.array.blocks[0].compute()
        assert arr.shape == jp2_raster_interface.array.blocks[0].shape
        jp2_raster_interface.array = None

class TestBinaryMaskReader:
    pass

class TestLabelledMaskReader:
    pass

class TestImageWriter:
    pass

class TestBinaryMaskWriter:
    pass

class TestLabelledMaskWriter:
    pass

# --- shp init
@pytest.mark.skip
def test_add_id_column_to_shp_file():
    pass

class TestBinaryMaskReader:
    pass

class TestLabelledMaskReader:
    pass

# -- fiona

class TestBinaryMaskShapeReader:
    @pytest.mark.skip
    def test_read_array_from_shapefile(self):
        pass

# -- ogr

class TestShapeReader:
    @pytest.mark.skip
    def test_open_shapefile(self):
        pass
    @pytest.mark.skip
    def test_add_FID_column(self):
        pass

class TestBinaryMaskShapeReader:
    @pytest.mark.skip
    def test_read_array_from_shapefile(self):
        pass

class TestLabelledMaskShapeReader:
    @pytest.mark.skip
    def test_read_array_from_shapefile(self):
        pass

@pytest.mark.skip
def test_rasterise_shapefile():
    pass
