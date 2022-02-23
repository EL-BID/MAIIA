"""
aoi_tools.py

Various convenience functions for studying and benchmarking segmentation results on AoIs.

Mostly tools for manipulating geopandas dataframes, overlaying polygons with RGB and 
binary segmentation rasters, calculating metrics, producing plots and polygonising rasters.
"""
import json
import gc
import shutil
import shlex
import subprocess
from functools import partial
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import dask.array as da
import geopandas as gpd
import pyproj
import matplotlib.pyplot as plt
import xarray as xr
import fiona
import rasterio
import rasterio.mask
import rioxarray as rx
import geocube
import geocube.rasterize
import shapely
import cufflinks as cf
import skimage.filters
import einops as eo
import plotly.graph_objs as go
import plotly.offline as pyo
from skimage.exposure import histogram
from skimage.filters import threshold_local, threshold_otsu
from shapely.geometry import box, mapping
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial
from rioxarray.rioxarray import RasterArray
from rasterio.features import shapes
from plotly import tools 
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import gim_cv
import gim_cv.config as cfg
import gim_cv.metrics as metrics


def load_ground_truth_df(
    ground_truth_vector_file:Union[Path, str],
    id_col:str='id'
) -> Tuple[gpd.GeoDataFrame, pd.Series]:
    """
    Load a set of vector ground truth polygons into a geopandas dataframe.
    
    Missing polygons are discarded. Any with missing id_col values are
    filled with unique substitutes and their indices kept in the returned series.
    
    Parameters
    ----------
    ground_truth_vector_file:
        A file readable by geopandas containing vector data.
    id_col: optional
        String name of column (feature name) corresponding to the unique id of each polygon
        
    Returns
    -------
    (A geodataframe containing the vector shapes, a series of any missing polygon indices)
    """
    # read shpfile with buildings in AOIs
    df_gt = gpd.read_file(ground_truth_vector_file)
    # throw away buildings with no geometry
    df_gt = df_gt[~df_gt.geometry.isna()]
    # throw away z dimension
    df_gt.geometry = df_gt.geometry.map(lambda polygon: shapely.ops.transform(lambda x, y, z=None: (x, y), polygon))
    # assign unique ids to assign to NaN values and keep track of these
    max_id = df_gt[id_col].max()
    missing_id_indices = df_gt[df_gt[id_col].isna()].index
    n_nan = missing_id_indices.shape[0]
    fill_ids = pd.Series(
        data=np.arange(start=max_id, stop=max_id+n_nan, dtype=np.uint32),
        index=missing_id_indices
    )
    df_gt[id_col] = df_gt[id_col].fillna(fill_ids).astype(np.uint32)
    # check all rows now have an ID value
    assert ~df_gt[id_col].isna().any()
    return df_gt, missing_id_indices


def load_aoi_df(
    aoi_vector_file:Union[Path,str],
    crs:pyproj.crs.crs.CRS=None,
    add_colours:bool=True,
    add_probs:bool=True,
) -> gpd.GeoDataFrame:
    """
    Loads a vector file containing AoI polygons into geopandas and
    applies a particular CRS. Uses KML driver if need be.
    
    Parameters
    ----------
    aoi_vector_file:
        A vector file format readable by geopandas (shp, KML, gpkg etc) 
        containing AoI polygons
    crs, optional:
        The CRS in which the polygons are embedded
    add_colours:
        Bool flag, generates black RGB columns for each polygon
    add_probs:
        Bool flag, creates a column with prob_gt = 9 for all polygons.        
        
    Returns
    -------
    A geopandas dataframe containing the AoI geometries and default RGB
    plotting formatting (black background)
    
    """
    kwargs = {}
    if str(aoi_vector_file).endswith('.kml'):
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        kwargs.update(driver='KML')
    # Read KML file with AOI polygons
    df_aoi_zones = gpd.read_file(aoi_vector_file, **kwargs)
    # throw away z dimension
    df_aoi_zones.geometry = df_aoi_zones.geometry.map(
        lambda polygon: shapely.ops.transform(lambda x, y, z=None: (x, y), polygon)
    )
    # apply CRS
    if crs:
        df_aoi_zones.to_crs(crs, inplace=True)
    assert df_aoi_zones.crs is not None, "Set a CRS!"
    # create black RGB attributes for background and assign building probability 0 everywhere
    if add_colours:
        df_aoi_zones[['r_plot', 'g_plot', 'b_plot']] = np.array((0, 0, 0), dtype=np.float32)
    if add_probs:
        df_aoi_zones['prob_gt'] = 0
    return df_aoi_zones


def join_aoi_data(
    df_bldgs:gpd.GeoDataFrame,
    df_aois:gpd.GeoDataFrame,
    add_colours:bool=True,
    add_probs:bool=True,
) -> gpd.GeoDataFrame:
    """
    Produces a geopandas dataframe of building polygons with AOI info
    
    Assigns additional columns (assigning building probability 1 to polygons)
    and generates random RGB plot colours for different buildings.
    
    Parameters
    ----------
    df_bldgs:
        A geopandas dataframe containing building polygons as rows
    df_aois:
        A geopandas dataframe containing AOI polygons as rows
    add_colours:
        Bool flag, generates random RGBs for each polygon
    add_probs:
        Bool flag, creates a column with prob_gt = 1 for all polygons.
        
    Returns
    -------
    A geopandas dataframe containing building polygons together with 
    the corresponding AoI columns, with randomly assigned colours to 
    distinguish buildings.
    
    """
    # spatial join to identify buildings with their aois
    df_gt_bldgs_in_aoi = gpd.sjoin(df_bldgs, df_aois, how='inner', op='within')
    # assign random rgb values to buildings
    if add_colours:
        df_gt_bldgs_in_aoi[['r_plot', 'g_plot', 'b_plot']] = (#np.round(
            np.random.rand(df_gt_bldgs_in_aoi.shape[0], 3) # * 255
        )#).astype(np.uint8)
    # assign building probability 1 to buildings
    if add_probs:
        df_gt_bldgs_in_aoi['prob_gt'] = 1
    return df_gt_bldgs_in_aoi.drop('index_right', axis=1)


def join_building_statistics(
    df_aoi:gpd.GeoDataFrame,
    df_bldgs:gpd.GeoDataFrame,
    aoi_name_col:str='Name',
    count_col:str='buildings_in_AOI'
) -> gpd.GeoDataFrame:
    """
    Adds total building counts to a geodataframe of AOI polygons
    
    Parameters
    ----------
    df_aoi:
        A geopandas dataframe of AOI polygons
    df_bldgs:
        A geopandas dataframe of buildings
    aoi_name_col:
        The column containing the AOI name
    count_col:
        The column name to add to the AOI dataframe containing the counts
        
    Returns
    -------
    A geopandas dataframe of AOI polygons supplemented by each building count
    
    """
    # derive per-aoi statistics
    if df_aoi.index.name != 'Name':
        df_aoi.set_index(aoi_name_col, inplace=True)
        df_aoi[count_col] = df_bldgs[aoi_name_col].value_counts()
        df_aoi.reset_index(inplace=True)
    return df_aoi


def derive_aoi_backgrounds(
    df_aoi:gpd.GeoDataFrame,
    df_bldgs:gpd.GeoDataFrame,
    id_col:str='id',
    gt_prob_col:str='prob_gt',
    bkgd_id_value:int=-1
) -> gpd.GeoDataFrame:
    """
    Creates a geopandas dataframe containing the backgrounds for a set of AoIs
    
    These are defined by taking the difference of the foreground (building polygons)
    and the AoIs themselves.
    
    Parameters
    ----------
    df_aoi:
        A geopandas dataframe containing AoI polygons
    df_bldgs:
        A geopandas dataframe containing building polygons
    id_col:
        The column name to which to assign a unique id for the background
    bkgd_id_value:
        The unique id to assign to the background geometries
    
    Returns
    -------
    A geopandas dataframe of the background geometries for each AoI
    """
    # cut the buildings out of the AOIs to get background polygons
    df_aoi_bkgd = gpd.overlay(df_aoi, df_bldgs, how='difference')
    # add background ground truth probability of 0 across whole polygon
    df_aoi_bkgd[gt_prob_col] = 0
    df_aoi_bkgd[id_col] = bkgd_id_value
    return df_aoi_bkgd


def threshold_otsu_nan(image:np.ndarray, nbins:int=256) -> float:
    """
    Return threshold value based on Otsu's method.
    
    Modified to ignore NaNs.
    
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
        
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    
    """
    if image.ndim > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warn(msg.format(image.shape))
        
    image_flat = image.ravel()
    # get rid of nans
    image_flat = image_flat[~np.isnan(image_flat)]
    
    # Check if the image is multi-colored or not
    first_pixel = image_flat[0]
    if np.all(image == first_pixel):
        return first_pixel

    hist, bin_centers = histogram(image_flat, nbins, source_range='image')
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def mod_threshold_otsu(arr:np.ndarray, frac:float=1.25, nbins:int=256) -> float:
    """
    Otsu modified by a factor (for example 1.25 raises the 
    threshold to be more conservative)
    
    Parameters
    ----------
    arr:
        An (M, N) numpy or dask array of raw float segmentation values
    frac: optional
        A multiplier to apply to the threshold (>1 == stricter)
    nbins: optional
        The number of bins to use in the histogram which derives the threshold
        
    Returns
    -------
    Derived float segmentation threshold
    """
    return frac*threshold_otsu_nan(arr, nbins=nbins)


def compute_array(arr:Union[da.Array, np.ndarray]) -> np.ndarray:
    """
    Evaluates a dask array and returns the result.
    Leaves numpy arrays unchanged.
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, da.Array):
        return arr.compute()
    
    
def build_aoi_dataset(
    rgb_raster:Union[Path, str],
    pred_raster:Union[Path, str],
    df_bldgs:gpd.GeoDataFrame,
    df_aoi:gpd.GeoDataFrame,
    year:Union[int, str]=None,
    aoi_name_col:str='Name',
    metrics_to_collect:Tuple[str] = (
        'jaccard_index',
        'tversky_index',
        'recall',
        'precision',
        'specificity',
        'npv'
    ),
    threshold_fn:callable=partial(mod_threshold_otsu, frac=1.)
) -> Tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    """
    Assimilates all layers of information about an AOI into an xarray Dataset
    
    Also produces and returns metrics calculated using the segmentation and GT.
    If functions with the metric names passed are present in the `metrics` module,
    they will be calculated and inserted into the resulting dataframe
    
    The output dataset contains:
        - RGB pixel values
        - raw predicted segmentation outputs
        - thresholded binary segmentation outputs
        - building ids assigned to pixel-wise locations
        - name of the AoI
        - year of the AoI capture (just a tag - can be e.g. 2019_winter)
        - a generated RGB colour for each different building for plotting
    
    Parameters
    ----------
    rgb_raster:
        Path to RGB raster/VRT containing the AoI
    pred_raster:
        Path to raw segmentation raster/VRT containing the AoI
    df_bldgs:
        Dataframe containing building polygons in the AoI
    df_aoi:
        Dataframe containing AoI polygons
    year: optional
        A year tag to assign to resulting dataset
    aoi_name_col: optional
        The name of the column containing the AoI name
    metrics_to_collect: optional
        A list or tuple of metrics to calculate (string funcion names in metrics.py)
    threshold_fn: optional
        A function which returns a segmentation threshold when applied to an
        (M,N) raw segmentation array
        
    Returns
    -------
    (An xarray dataset with the layers described above,
     dataframe of pre(,post) segmentation statistics computed against ground truth)
    """
    assert df_aoi.shape[0] == 1, "df_aoi should have exactly one row"
    # open image raster, predicted mask VRTs with rioxarray
    xds_rgb = rx.open_rasterio(rgb_raster, chunks=(-1, 5120, 5120))
    xds_pred = rx.open_rasterio(pred_raster, chunks=(-1, 5120, 5120))
    # prepare dataframe for collecting segmentation metrics per AoI
    df_seg_metrics = pd.DataFrame(columns=metrics_to_collect)
    df_seg_metrics_thresh = pd.DataFrame(columns=metrics_to_collect)
    # subtract buildings from backgrounds to collect background statistics
    df_aoi_bkgds = derive_aoi_backgrounds(df_aoi, df_bldgs)
    row = df_aoi.iloc[0]
    # create one geocube for AoI with all necessary layers
    aoi = row[aoi_name_col]
    minx, miny, maxx, maxy = row.geometry.bounds
    # first clip_box on the large predicted mask VRT 
    # doing clip directly is apparently terribly optimised and loads everything into memory
    aoi_pred_array = xds_pred.rio.clip_box(*row.geometry.bounds)
    aoi_pred_array = aoi_pred_array.rio.clip(geometries=[row.geometry], crs=df_bldgs.crs)
    # ditto for RGB orthos
    aoi_rgb = xds_rgb.rio.clip_box(*row.geometry.bounds)
    aoi_rgb = aoi_rgb.rio.clip(geometries=[row.geometry], crs=df_bldgs.crs)
    # and for GT arrays
    cube_gt = make_geocube(
        df_bldgs.query(f"{aoi_name_col} == '{aoi}'"),
        measurements=['id', 'r_plot', 'g_plot', 'b_plot', 'prob_gt'],
        like=aoi_rgb,
        rasterize_function=partial(geocube.rasterize.rasterize_image, all_touched=True)
    )
    # burn in zero values for RGB inside full AOI polygons
    cube_aoi = make_geocube(
        df_aoi.query(f"{aoi_name_col} == '{aoi}'"),
        measurements=['r_plot', 'g_plot', 'b_plot', 'prob_gt'],
        like=aoi_rgb,
        rasterize_function=partial(geocube.rasterize.rasterize_image, all_touched=True)
    )
    # get probabilities in background of AOIs
    cube_bkgd = make_geocube(
        df_aoi_bkgds.query(f"{aoi_name_col} == '{aoi}'"),
        measurements=['id'],
        like=aoi_rgb,
        rasterize_function=partial(geocube.rasterize.rasterize_image, all_touched=True)
    )
    # create final cube from unique bands
    cube = cube_gt + cube_aoi + cube_bkgd
    # add AoI
    cube['key'] = aoi.lower().replace(' ', '_')
    # add year info
    if year is not None:
        cube['year'] = year
    # burn in id values for bldgs and backgrounds
    cube['AoI'] = aoi
    cube['mask'] = (('y','x'), np.isnan(cube_aoi['prob_gt'].data))
    # add ids from different layers, ignoring nans
    cube['id'] = cube_bkgd['id']
    cube['id'].data = np.nansum(np.dstack((cube_bkgd['id'], cube_gt['id'])), axis=-1).astype(np.int32)
    # gt
    cube['prob_gt'] = (cube_aoi['prob_gt'] + cube_gt['prob_gt']).fillna(0.)
    # encode RGB values by overlapping background and gt
    cube['r_plot'] = cube_aoi['r_plot'].fillna(1.) + cube_gt['r_plot'].fillna(0.)
    cube['g_plot'] = cube_aoi['g_plot'].fillna(1.) + cube_gt['g_plot'].fillna(0.)
    cube['b_plot'] = cube_aoi['b_plot'].fillna(1.) + cube_gt['b_plot'].fillna(0.)
    # add layer for predicted probabilities
    cube['pred'] = (('y', 'x'), aoi_pred_array.squeeze())
    # ortho RGB value layers
    cube['r'] = (('y','x'), aoi_rgb[0])
    cube['g'] = (('y','x'), aoi_rgb[1])
    cube['b'] = (('y','x'), aoi_rgb[2])
    # calculate metrics
    df_seg_metrics.loc[aoi] = [
        getattr(metrics, m)(
            compute_array(cube['prob_gt'].data), compute_array(cube['pred'].data)
        )
        for m in metrics_to_collect
    ]
    # evaluate threshlded predictions and add layer
    comp_arr = compute_array(cube.pred.data)
    thresh = threshold_fn(comp_arr)
    print(f"using threshold: {thresh}")
    cube['pred_thresh'] = (('y','x'), (comp_arr > thresh).astype(np.uint8))
    # collect segmentation metrics per AoI with threshold
    df_seg_metrics_thresh.loc[aoi] = [
        getattr(metrics, m)(cube['prob_gt'].data, cube['pred_thresh'].data)
        for m in metrics_to_collect
    ]
    return cube, df_seg_metrics, df_seg_metrics_thresh


def plot_aoi_segmentation_and_gt(
    cube:xr.Dataset,
    gt_attr:str='prob_gt',
    pred_attr:str='pred',
    mask_attr:str='mask',
    rgb_attrs:List[int]=list('rgb'),
    gt_rgb_attrs:List[int]=['r_plot','g_plot','b_plot'],
    save_to:Union[Path, str]='.',
    show:bool=True,
    fname:str='segmentation_and_gt',
    append_fname_attrs:List[str]=['year', 'key'],
    figsize:Tuple[int]=(20, 20),
    buildings_coloured=True,
    dpi:int=100,
) -> plt.Figure:
    """
    Plots RGB image, raw segmentation and ground truth together
    
    Optionally saves output figure to file.
    
    Parameters
    ----------
    cube:
        An xarray dataset generated by build_aoi_dataset
    gt_attr: optional
        The name of the layer containing the ground truth values [e.g. 0 or 1]
    pred_attr: optional
        The name of the layer containing the predicted values [between 0. and 1.]
    rgb_attrs: optional
        The names of the layers containing the R, G and B values of the raster
    gt_rgb_attrs: optional
        The names of the layers containing the R, G and B values of the ground truth
        (this is purely arbitrary and aesthetic, and serves to distinguish different 
        buildings)
    save_to: optional
        A target directory in which to save generated images
    show: optional
        Flag to show plot when function is run
    figsize: optional
        figure size passed to plt.subplots
    fname: optional
        Base filename (without extension)
    append_fname_attrs: optional
        A list of attributes to append to the base filename if present in cube/dataset
    dpi: optional
        DPI for saved png plot 
        
    Returns
    -------
    matplotlib figure
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    plt.gray()
    axs[0].imshow(cube[rgb_attrs].to_array().data.transpose(1,2,0))
    axs[1].imshow(cube[pred_attr].data)
    if buildings_coloured:
        gt_arr = cube[gt_rgb_attrs].to_array().data.transpose(1,2,0)
    else:
        gt_arr = np.all(cube[gt_rgb_attrs].to_array().data.transpose(1,2,0) != [0.,0.,0.], axis=-1).astype(np.uint8)
    axs[2].imshow(gt_arr)
    plt.tight_layout()
    if show:
        plt.show()
    if save_to:
        for attr in append_fname_attrs:
            if hasattr(cube, attr):
                fname += f'_{attr}_{cube[attr].data}'
        fname +='.png'
        path = Path(save_to) / Path(fname)
        plt.savefig(path, dpi=dpi)
    return fig



def zonal_stats_from_cube(
    cube:xr.Dataset,
    id_attr:str='id',
    pred_attr:str='pred',
    aoi_name_attr:str='AoI'
) -> pd.DataFrame:
    """
    Calculate zonal statistics (per polygon: so per-building) over an 
    xarray dataset (where unique buildings are specified by the id 
    attribute)
    
    Calculates mean/min/max/stdev probability predicted for each building
    footprint
    
    Parameters
    ----------
    cube:
        An XArray Dataset containing predicted probabilities and building ids
    id_attr:
        The dataset attribute containing the building ids
    pred_attr:
        The dataset attribute containing the raw segmentation predictions
    aoi_name_attr:
        The dataset attribute containing the AoI name
    """
    # calculate zonal statistics (per building)
    grouped_probas = cube.drop([
            k for k in cube.data_vars.keys() if k not in (id_attr, pred_attr, aoi_name_attr)
    ]).groupby(cube[id_attr])
    grid_mean = grouped_probas.mean().rename({pred_attr: f"{pred_attr}_mean"})
    grid_min = grouped_probas.min().rename({pred_attr: f"{pred_attr}_min"})
    grid_max = grouped_probas.max().rename({pred_attr: f"{pred_attr}_max"})
    grid_std = grouped_probas.std().rename({pred_attr: f"{pred_attr}_std"})
    zonal_stats = xr.merge([grid_mean, grid_min, grid_max, grid_std])
    df = zonal_stats.to_dataframe()
    return df.drop(['spatial_ref'], axis=1)



def confusion_map(
    cube:xr.Dataset,
    gt_attr:str='prob_gt',
    pred_attr:str='pred_thresh',
    mask_attr:str='mask',
    rgb_attrs:List[int]=list('rgb'),
    save_to:Union[str, Path]='.',
    show:bool=True,
    figsize:Tuple[int]=(24, 36),
    fname:str='confusion_map',
    append_fname_attrs:List[str]=['year', 'key'],
    dpi:int=100
) -> plt.Figure:
    """
    Produces a confusion map from an xarray dataset.
    
    This must contain layers containing predicted probabilities 
    in [0, 1] and ground truth probabilities in {0, 1}, along 
    with an optional mask layer marking the pixels not included.
    
    Optionally displays the plot and saves it to a directory.
    
    Parameters
    ----------
    cube:
        An XArray dataset with at least two layers: predicted and ground truth
    gt_attr: optional
        The name of the layer containing the ground truth values [e.g. 0 or 1]
    pred_attr: optional
        The name of the layer containing the thresholded predicted values [0 or 1]
    mask_attr: optional
        The name of the layer containing the mask values [I think True/False - check]
    rgb_attrs: optional
        The names of the layers containing the R, G and B values of the raster
    save_to: optional
        A directory in which to save output images as .png
    show: optional
        Flag to show plot when function is run
    figsize: optional
        figure size passed to plt.subplots
    fname: optional
        Base filename (without extension)
    append_fname_attrs: optional
        A list of attributes to append to the base filename if present in cube/dataset
    dpi: optional
        DPI for saved png plot 
        
    Returns
    -------
    plt.Figure:
        The confusion map plot
    """
    # true positives - colour white
    tp = cube[pred_attr].data * cube[gt_attr].data
    tp = tp[:,:,np.newaxis] * np.array((1., 1., 1.)) # white
    # false positives - colour red
    fp = cube[pred_attr].data * (1-cube[gt_attr].data)
    fp = fp[:,:,np.newaxis] * np.array((1., 0, 0))
    # false negatives - colour green
    fn = (1-cube[pred_attr].data) * cube[gt_attr].data
    fn = fn[:,:,np.newaxis] * np.array((0, 1., 0))
    # true negatives - colour black
    tn = (1-cube[pred_attr].data) * (1-cube[gt_attr].data)
    tn = tn[:,:,np.newaxis] * np.array((0., 0., 0.))
    # mask the pixels which are not in the polygon
    if hasattr(cube, mask_attr):
        plt_arr = np.ma.masked_array(
            tp+fp+fn+tn,
            mask=eo.repeat(cube[mask_attr].data, 'h w -> h w c', c=3)
        )
    else:
        plt_arr = fp+fp+fn+tn
    # create axes and optionally show map
    plt_args = (1, 2) if rgb_attrs else (1, )
    fig, axs = plt.subplots(*plt_args, figsize=figsize)
    # optionally plot the raw RGB image alongside the confusion map
    plt_ix = 0
    if rgb_attrs:
        axs[plt_ix].imshow(cube[rgb_attrs].to_array().data.transpose(1,2,0))
        plt_ix+=1
    axs[plt_ix].imshow(plt_arr)
    plt.tight_layout()
    if show:
        plt.show()
    # optionally save to fname, appending e.g. year and AoI key data from cube
    # under directory save_to
    if save_to:
        for attr in append_fname_attrs:
            if hasattr(cube, attr):
                fname += f'_{attr}_{cube[attr].data}'
        fname +='.png'
        path = Path(save_to) / Path(fname)
        plt.savefig(path, dpi=dpi)
    return fig



def polygonise(
    cube:xr.Dataset,
    pred_attr:str='pred_thresh',
    aoi_name_attr:str="AoI"
) -> gpd.GeoDataFrame:
    """
    Creates a geopandas dataframe of polygons from a layer of an XArray dataset.
    
    Parameters
    ----------
    cube:
        An Xarray dataset containing a binary segmentation array as a layer 
    pred_attr:
        The name of the attribute/layer of the Dataset with the binary raster
    aoi_name_attr:
        Name of the dataset attribute containing the AoI name
    Results
    -------
    """
    aoi = str(cube[aoi_name_attr].data)#"key"
    print("Running polygonisation for: ", aoi)
    results = [
        {'properties': {'raster_val': v}, 'geometry': s}
        for (s, v) in 
            shapes(
                cube[pred_attr].data,
                mask=cube[pred_attr].data==1,
                transform=cube[pred_attr].rio.transform()
            )
    ]
    df_polygonised = gpd.GeoDataFrame.from_features(results, crs=cube.spatial_ref.crs_wkt)
    df_polygonised[aoi_name_attr] = aoi
    print(df_polygonised.shape[0], "shapes found")
    return df_polygonised  



def calculate_tile_extents(tile_paths:List[Union[Path, str]]) -> gpd.GeoDataFrame:
    """
    Build a geopandas dataframe containing the polygons spanning a list of tiles
    
    Useful for finding subset of rasters which intersect some AoIs
    
    Parameters
    ----------
    tile_paths:
        List of paths to rasters which tile a region
        
    Returns
    -------
    A geopandas dataframe of the tile polygons along with their paths
    """
    extents = [rx.open_rasterio(p).extent for p in tile_paths]
    return gpd.GeoDataFrame(
        data=[str(p) for p in tile_paths],
        columns=['path'],
        geometry=extents
    )

## tools for polygon metrics

def plot_polygonised(
    df_true,
    df_pred,
    aoi_ix=1,
    true_poly_attr="Name",
    pred_poly_attr="AoI",
    true_poly_attr_fmt="Polygone {aoi_ix}",
    pred_poly_attr_fmt="polygone_{aoi_ix}",
    year='2018',
    resolution='20cm',
    save=True
):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].set_title('Ground truth')
    qtrue = f'{true_poly_attr} == "{true_poly_attr_fmt.format(aoi_ix=aoi_ix)}"'
    df_true.query(qtrue).plot(ax=axs[0])
    axs[1].set_title(f'Polygonised {resolution} {year}')
    qpred = f'{pred_poly_attr} == "{pred_poly_attr_fmt.format(aoi_ix=aoi_ix)}"'
    df_pred.query(qpred).plot(ax=axs[1])
    if save:
        plt.savefig(f'./polygonised_comparison_aoi_{aoi_ix}_{year}_{resolution}.png')
        
        
def add_intersection(
    df,
    col1='geometry',
    col2='pred_geometry',
    target_column='geometry_intersection',
    area_column='area_intersection',
    frac_column='frac_intersection',
    buffer=0.0
):
    df[target_column] = df.apply(
        (
            lambda row: row[col1].buffer(buffer).intersection(row[col2].buffer(buffer))
            if (row[col1] and row[col2]) else np.nan
        ),
        axis=1
    )
    df[target_column] = gpd.GeoSeries(df[target_column])
    df[area_column] = df[target_column].area
    df[frac_column] = df[area_column] / df[col1].area
    return df


def add_union(
    df,
    col1='geometry',
    col2='pred_geometry',
    target_column='geometry_union',
    area_column='area_union',
    frac_column='frac_union',
    buffer=0.0
):
    df[target_column] = df.apply(
        (
            lambda row: row[col1].buffer(buffer).union(row[col2].buffer(buffer))
            if (row[col1] and row[col2]) else np.nan
        ),
        axis=1
    )
    df[target_column] = gpd.GeoSeries(df[target_column])
    df[area_column] = df[target_column].area
    df[frac_column] = df[area_column] / df[col1].area
    return df


def add_intersection_and_union(
    df,
    col1='geometry',
    col2='pred_geometry'
):
    _df = add_union(add_intersection(df, col1, col2), col1, col2)
    _df['IoU'] = _df['area_intersection'] / _df['area_union']
    return _df


def exactly_one_intersecting(
    df,
    left_id='id_pred',
    right_id='id_true'
):
    _df = df[~df[right_id].isna()]
    return _df.set_index(left_id)[df[left_id].value_counts() == 1].reset_index()


def many_intersecting(
    df,
    left_id='id_pred',
    right_id='id_true'
):
    _df = df[~df[right_id].isna()]
    return _df.set_index(left_id)[df[left_id].value_counts() > 1].reset_index()


def none_intersecting(
    df,
    left_id='id_pred',
    right_id='id_true'
):
    return df[df[right_id].isna()].reset_index()


def plot_iou_distn_per_aoi(
    df,
    groupby='AoI',
    title_suffix='2018, 1m'
):
    gb = df.groupby(groupby)
    hists = {}
    df = pd.DataFrame()
    for k in gb.groups.keys():
        _df = gb.get_group(k)
        counts, bins = np.histogram(_df['IoU'], bins=np.arange(0, 1.1, 0.1))
        df[k] = counts
        bins = 0.5 * (bins[:-1] + bins[1:])
    else:
        df.index = bins
    data= go.Heatmap(
        x=df.columns,
        y=df.index,
        z=(df / df.sum(axis=0)).values,
    )
    fig = go.Figure(data=data)
    fig.update_layout(title=f'Distribution of IoU for single-polygon matches per AoI ({title_suffix})')
    return fig


def plot_iou_one_intersecting(df, aoi_ix=1, resolution='1m', year='2018', column='raster_val', save=True):
    fig, axs = plt.subplots(1, 2, figsize=(16,16))
    aoi_name = f'Polygone {aoi_ix}'
    aoi_name_lower = aoi_name.lower().replace(' ', '_')
    plt.tight_layout()
    # Ground truth
    axs[0].set_title('Ground Truth')
    df[(df['Name'] == aoi_name)].plot(
        ax=axs[0],
        column=column,
        legend=True,
        legend_kwds={
            'label' : 'Building probability',
            'orientation' : 'horizontal'
        }
    )

    # Polygonisation with IoU
    axs[1].set_title(f'Polygonisation (DeepResUNet, Otsu thresholding * 1.25, {resolution}, {year})')
    df[(df['Name'] == aoi_name)].set_geometry('pred_geometry').plot(
        column='IoU',
        ax=axs[1],
        legend_kwds={
            'label': "IoU",
            'orientation': "horizontal"
        },
        legend=True
    )
    if save:
        plt.savefig(f'./IoU_one_intersecting_{resolution}_{year}_{aoi_name_lower}.png')
        
        
def plot_processed_polygons(
    df_true,
    df_pred,
    aoi_ix=1,
    resolution='1m',
    year='2018',
    save=False,
    save_to='.',
    left_title='Ground Truth',
    right_title='Predicted polygons',
    aoi_col_true='Name',
    aoi_col_pred='AoI',
    figsize=(20, 10)
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    aoi_name = f'Polygone {aoi_ix}'
    aoi_name_lower = aoi_name.lower().replace(' ', '_')
    plt.tight_layout()
    # Ground truth
    axs[0].set_title(f'{left_title} ({resolution}, {year})')
    df_true[(df_true[aoi_col_true] == aoi_name)].plot(
        ax=axs[0],
    )
    # Polygonised segmentation
    axs[1].set_title(f'{right_title} ({resolution}, {year})')
    df_pred[(df_pred[aoi_col_pred] == aoi_name_lower) | (df_pred[aoi_col_pred] == aoi_name)].plot(
        column='status',
        ax=axs[1],
        legend=True
    )
    if save:
        plt.savefig(
            str(Path(save_to) / Path(f'./processed_polygons_{resolution}_{year}_{aoi_name_lower}.png')),
            dpi=200
        )
    return fig