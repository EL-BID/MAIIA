import logging

import fiona
from satproc.utils import reproject_shape
from shapely.geometry import shape
from tqdm import tqdm

from maiia.utils import get_epsg_utm_from

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def filter_by_min_area(src_file, dst_file, min_area=500, utm_code=None):
    if not utm_code:
        utm_code = get_epsg_utm_from(src_file)
        print(f"Using projected CRS {utm_code} for filtering by meters")

    with fiona.open(src_file) as src:
        with fiona.open(dst_file, "w", driver="GPKG", crs=src.crs, schema=src.schema) as dst:
            for feature in tqdm(src, desc=f"Filtering polygons by area (>={min_area}m)", ascii=True):
                geom = shape(feature['geometry'])
                repr_geom = reproject_shape(geom, src.crs, utm_code)
                if repr_geom.area >= min_area:
                    dst.write(feature)
