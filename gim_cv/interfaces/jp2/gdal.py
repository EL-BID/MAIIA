import dask.array as da
import dask

import numpy as np
import dask.array as da
import timbermafia as tm


import logging

log = logging.getLogger(__name__)



def ds_to_dask_array(
    image_ds,
    window_size=(224, 224),
    target_block_size_mb=100,
    virtual_mem_threshold_mb=4000
) -> dask.array.Array:
    """ 
    Loads a gdal image dataset as a dask array. 
    
    Attempts to come up with a decent block size and shape, being an 
    integer multiple of some window dimensions and around 
    target_block_size MB

    Parameters
    ----------
    image_ds : 
        a gdal image dataset
    window_size : 
        a tuple (n_x_pixels, n_y_pixels)
    target_block_size_mb : 
        integer number of megabytes for each block
    virtual_mem_threshold_mb : 
        integer size which array has to be before using virtual memory 
        loader (as opposed to loading whole img into memory as np.array)

    Returns
    -------
    a dask array (row, col, channel)

    To do
    -----
    sample_input_image.GetRasterBand(1).GetBlockSize() -> [1024,1024]
    the "natural" size to access the dataset. can incorporate this?
    https://gdal.org/doxygen/classGDALRasterBand.html
    #af2ee6fa0f675d7d52bc19f826d161ad6
    """
    sample_image_arr = image_ds.GetVirtualMemArray()
    # try to derive an optimal chunk size which is a multiple of patch size
    image_size_mb = sample_image_arr.nbytes/(1024*1024)
    log.debug(f"Processing array of size: {image_size_mb:.2f} MB")
    log.debug(f"With dimensions: {sample_image_arr.shape}")
    n_blocks_ideal = round(image_size_mb/target_block_size_mb)
    window_size_x, window_size_y = window_size
    n_patch_x, n_patch_y = (image_ds.RasterXSize/window_size_x,
                            image_ds.RasterYSize/window_size_y)
    # the optimal chunk size (if we take them to be square) is a multiple k of
    # the patch size such that the number of rescaled x patches * y patches
    # is close to the ideal number of blocks. we round it to ensure integer.
    k_opt = round(np.sqrt(n_patch_x * n_patch_y / n_blocks_ideal))
    new_window_size_x, new_window_size_y = (int(k_opt * window_size_x),
                                          int(k_opt * window_size_y))
    log.debug("Partitioning image into dask array with chunks of size: "
          f"({new_window_size_x}, {new_window_size_y})")
    # this takes some time
    t0 = pc()
    if image_size_mb > virtual_mem_threshold_mb:
        log.debug("Loading array from virtual memory into dask array: "
              "This can take several minutes! (1.8GB => ~18m)")
        _arr = sample_image_arr.transpose(1, 2, 0)
    else:
        log.debug("Loading image into in-memory numpy array and converting to dask."
              " This can take a few minutes (1.8GB => ~6m).")
        _arr = image_ds.ReadAsArray().transpose(1, 2, 0)
    # workaround to avoid blowing up RAM:
    # see https://github.com/dask/dask/issues/5601
    ws = new_window_size_x
    # divide into sub-arrays via simple slicing along the first dim
    sliced = [da.from_array(_arr[slice(ind * ws, (ind * ws) + ws )])
              for ind in range(1 + _arr.shape[0]//ws)]
    # stack the sliced dask arrays and concatenate them into a single one
    dask_arr = da.concatenate(sliced).rechunk(
        (new_window_size_y, new_window_size_x, 3)
    )
    #dask_arr = da.from_array(_arr, chunks=...)
    log.debug(f"Dask array created. Took {pc()-t0:.2f}s")
    return dask_arr
