Inference
=========

The :mod:`~gim_cv.inference` module is responsible for providing classes which 
integrate datasets and preprocessing pipelines into a simple object which can be 
fed a ``tf.keras`` segmentation model which will run inference with the model's 
``predict`` method and write the results to disk in the form of ``.tif`` rasters.

The main classes playing this role are :class:`~gim_cv.inference.InferenceDataset` and 
:class:`~gim_cv.inference.CompositeInferenceDataset`. These can be used standalaone 
(by specifying which image files to create from directly) or be created directly 
from a :class:`~gim_cv.datasets.Dataset` object (comprised of RGB images). 
We will cover both of these cases in turn.

For the very short version, look at :ref:`load_inference_data`.

Inference Datasets
------------------

An :class:`~gim_cv.inference.InferenceDataset` converts *one* corresponding 
image file into preprocessed array of patches, facilitates running segmentation on 
these, and handles reassembling and writing of the results to a raster file. 
It accepts the following main arguments:

* Path to the image source file (ideally in ``.tif`` format)
* Path to the target mask file which will be written by the model (also ``.tif``) - optional
* A function which, when called, creates a preprocessing pipeline for the image - optional

There are also various options for controlling pruning functions to eliminate 
invalid arrays (such as empty arrays with all white or all black pixels). See the 
:class:`~gim_cv.inference.InferenceDataset` API documentation for more details.

Once the :class:`~gim_cv.inference.InferenceDataset` has been created, its 
:meth:`~gim_cv.inference.InferenceDataset.prepare` method must be called (this will 
create an instance of the necessary preprocessing pipelines, then create the Dask 
task graph corresponding to loading the arrays from the source files and performing 
all of the preprocessing operations). Once the prepare method has been called the 
image patch dask array will be accessible as the attribute ``X``. 

After this stage, the inference dataset object will have access to the method
:meth:`~gim_cv.inference.InferenceDataset.schedule_inference`,  which accepts a 
model and optionally a directory to write the output segmented rasters to (this 
is ignored if an explicit ``mask_tar`` is set). This will create the dask array 
``y`` with the segmentation results (but not yet compute them). This array 
has a chunk size of 1 (patch) which is fixed to equal the inference batch size.

Once this has been run with a provided model, the actual inference and writing of the 
results to disk in raster format can be triggered with the 
:meth:`~gim_cv.inference.InferenceDataset.write_mask_raster` method. This will compute 
the segmentation mask patches, reassemble them into the input raster's shape and 
write out to the target ``.tif`` file. It also accepts a boolean ``overwrite`` 
parameter which one can toggle to skip running inference for rasters already processed 
by this model (i.e. if the target file already exists).

Here's an example::

    from functools import partial
    from pathlib import Path
    from gim_cv.inference import InferenceDataset
    from gim_cv.preprocessing import get_image_inference_pipeline

    # create inference dataset
    # can either specify the output file path (mask_tar) directly here or omit it
    # if omitted, can pass instead "output_directory" to the "schedule_inference" method
    # this will borrow the name of the input raster and append the model name and uuid + . tif
    ids = InferenceDataset(
        image_src = Path('/path/to/my/img.tif'),
        mask_tar = Path('/path/to/output/mask.tif'),
        image_pipeline_factory=partial(
            get_image_inference_pipeline,
            inference_window_size=1024
        ) # example overriding default pipeline
    )
    # extract arrays, create dask graph for preprocessing etc
    ids.prepare()
    # build dask array for segmentation results by passing a model (e.g. a Segmentalist)
    ids.schedule_inference(model=model) # can specify 'output_directory' here if 'mask_tar' not fixed
    # calculate segmentation and write results
    ids.write_mask_raster()

Composite Inference Datasets
----------------------------

Most of the time you will want to run inference with a model on multiple image 
files, from a :class:`~gim_cv.datasets.Dataset` comprised of multiple files and/or 
from multiple :class:`~gim_cv.datasets.Dataset` objects (for example, of different 
areas and/or the same areas over multiple years).

The :class:`~gim_cv.inference.CompositeInferenceDataset` class is a thin wrapper for a set 
of individual :class:`~gim_cv.inference.InferenceDataset` objects which allows an easy 
interface to run inference on many files at once. 

A composite inference dataset can be created by just adding inference datasets to 
each other. It has the same API as :class:`~gim_cv.inference.InferenceDataset` (delegating 
to the constituents' methods) except for the plural form 
:class:`~gim_cv.inference.CompositeInferenceDataset.write_mask_rasters`.
Here's an example of creating one manually::

    from pathlib import Path
    from gim_cv.inference import InferenceDataset

    # skip providing mask_tar for each here and instead choose an "output_directory" below
    ids1 = InferenceDataset(image_src=Path('/path/to/img1.tif'))
    ids2 = InferenceDataset(image_src=Path('/path/to/img2.tif'))

    # create a CompositeInferenceDataset by adding together as many InferenceDatasets as you like
    ids_all = ids1 + ids2

    # create dask arrays
    ids_all.prepare()
    # pass model with which to run inference. specify a directory for outputs here
    ids.schedule_inference(model=model, output_directory=Path('/path/to/output_dir'))
    # calculate segmentation and write results
    ids.write_mask_rasters()

A list of the constituent :class:`~gim_cv.inference.InferenceDataset` objects are available 
via the ``constituents`` attribute.

.. _load_inference_data:

Creating inference dataset objects from predefined datasets
-----------------------------------------------------------

You can create a :class:`~gim_cv.inference.CompositeInferenceDataset` directly from a 
:class:`gim_cv.datasets.Dataset` object using the :meth:`gim_cv.datasets.Dataset.load_inference_data` 
method. This will construct a :class:`~gim_cv.inference.InferenceDataset` from each image 
defined in the :class:`gim_cv.datasets.Dataset`'s ``image_paths`` attribute, combine these 
and return a :class:`~gim_cv.inference.CompositeInferenceDataset`.

Like :meth:`~gim_cv.dataset.load_training_data`, :meth:`~gim_cv.dataset.load_inference_data` accepts 
arguments which allow one to choose the pipeline and hyperparameters thereof (such as patch size 
and batch size - see :doc:`preprocessing` documentation). See its API documentation for more details.

Here's an example::

    from gim_cv.datasets import get_dataset
    

    # create inference dataset
    ds = get_dataset('phil_man_14_50cm_01')
    # say we want to resample to 0.5m^2 to use a model trained at this resolution
    target_spatial_resolution = .5 
    inference_window_size = 896
    ids = ds.load_inference_data(
        resample_factor = ds.spatial_resolution / target_spatial_resolution,
        inference_window_size=inference_window_size
    )
    ids.prepare()

    # create a directory to store the results
    output_path = ds.image_paths[0].parent / Path('segmentation_outputs')
    output_path.mkdir(exist_ok=True, parents=True)
    # prepare inference with model
    ids.schedule_inference(
        model,
        output_directory=output_path
    )
    log.info("inference job scheduled")

    # run inference
    log.info("generating mask rasters...")
    ids.write_mask_rasters(overwrite=False)


Inference script
----------------

For an example inference script using the ``Segmentalist`` model with datasets predefined 
in the :mod:`gim_cv.datasets` module, you can check out the documentation, comments and 
help string for ``gim-cv/bin/inference/run_inference_segmentalist.py``.

This script requires one to select the training datasets and loss function used to train models, 
and locates the model in the trained models directory which matches these with the lowest 
validation loss. An inference patch size must be specified (default is currently 1024) along with 
an input dataset on which to run inference. The inference will then run and create mask rasters 
in a new subdirectory of the parent directory of the input dataset.

Here's an example use case, selecting the best model trained on the training datasets "phil_man_14_50cm_04" and 
"phil_man_14_50cm_05" with the weighted binary cross entropy loss function, and running inference with a patch size of 1024 
on the "phil_man_14_50cm_03" dataset::

    $ python run_inference_segmentalist.py -td phil_man_14_50cm_04,phil_man_14_50cm_05 -d phil_man_14_50cm_03 -w 1024 -l wbce_adaptive

The body of the script is very simple and relies on selecting the best trained model as described in 
:ref:`comparing trained models`::

    df_trained_models = collate_run_data(models_dir, model_name="Segmentalist")     
    df_sorted = df_trained_models.sort_values(by='lowest_val_loss').query(
        f'datasets == "{args.training_datasets}" and loss_fn == "{args.loss_fn}"'
    )
    best_row = df_sorted.iloc[0]
    model = Segmentalist.load_from_metadata(best_row)

Followed by a loop to create inference datasets for each requested::

    # run inference for each requested dataset sequentially
    for ds_tag in ds_tags:
        inference_window_size = args.window_size # implicit here, will need to feed explicitly again to model
        
        # create inference dataset
        ds = datasets.get_dataset(ds_tag)
        target_spatial_resolution = (ds.spatial_resolution if not args.target_spatial_resolution else args.target_spatial_resolution)
        ids = ds.load_inference_data(
            resample_factor = ds.spatial_resolution / target_spatial_resolution,
            inference_window_size=inference_window_size
        )
        ids.prepare()

        # save
        if not args.output_dir:
            output_path = ds.image_paths[0].parent / Path('seg_outputs')
        else:
            output_path = args.output_dir
        output_path.mkdir(exist_ok=True, parents=True)
        
        # prepare inference with model
        ids.schedule_inference(
            model,
            output_directory=output_path
        )

        # run inference
        ids.write_mask_rasters(overwrite=False)

Correcting unsatisfcatory segmentation results
----------------------------------------------

You may run into the situation where a segmentation model produces
poor results delineating certain difficult objects. There are currently 
two methods to rectify this situation which can be done independently or
combined.

Manually annotating ground truth data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first and simplest way is to provide an accurate ground-truth 
segmentation mask for the poorly-segmented objects and retrain your 
model with these included in the training data.

For example, suppose that you have observed serious errors in a test region 
`bad_AoI.tif` for which you currently do not have ground truth data. You 
can then manually draw the correct object boundaries in, for example, 
a shapefile `bad_AoI_ground_truth.shp`. You can then create a new dataset as::

    import gim_cv.datasets as datasets

    corrected_dataset = datasets.Dataset(
        tag='bad_AoI',
        spatial_resolution=0.2,
        image_paths = ['/path/to/bad_AoI.tif'],
        mask_paths = ['/path/to/bad_AoI_ground_truth.shp'])
    )

You can then follow the instructions to train a model including this additional 
dataset described in :doc:`training`.

Use a tile overlapping strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second method is to reduce edge effects present on prediction results. In this case, go to the main portal's directory `.notebooks/Portal`.

From the repository, you can launch both the training and inference process as specified above::

    $ cd notebooks/portal

    python Infer_train_simple.py
