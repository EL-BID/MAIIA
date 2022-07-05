import logging
import subprocess
import math

import fiona
from satproc.utils import reproject_shape
from shapely.geometry import shape

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def run_command(cmd):
    _logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)


def get_epsg_utm_from(vector_path):
    """Calculate UTM zone from a vector file in WGS84 geographic coordinates"""
    with fiona.open(vector_path) as src:
        some_feat = next(iter(src), None)
        if not some_feat:
            raise ValueError(f"{vector_path} has no features")
        some_geom = shape(some_feat["geometry"])
        if src.crs["init"] != "epsg:4326":
            some_geom = reproject_shape(some_geom, src.crs["init"], "epsg:4326")
        return get_epsg_utm_from_wgs_geom(some_geom)


def get_epsg_utm_from_wgs_geom(geom):
    """
    Calculate UTM zone from a geometry in WGS84 geographic coordinates and
    get corresponding EPSG code.

    """
    centroid = geom.centroid
    lon, lat = centroid.x, centroid.y
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = f"{utm_band}"
    if lat >= 0:
        epsg_code = f"epsg:326{utm_band}"
    else:
        epsg_code = f"epsg:327{utm_band}"
    return epsg_code
