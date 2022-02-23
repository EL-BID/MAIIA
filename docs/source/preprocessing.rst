Preprocessing
=============

The :mod:`~gim_cv.preprocessing` module contains the machinery needed 
to convert large raw raster arrays (extracted via the 
:mod:`~gim_cv.interfaces` module) into training- (or inference-) ready 
arrays. Its primary role is then to turn these large raster and mask arrays with 
shapes ``(H, W, C)`` and ``(H, W, C')`` into ``N`` corresponding small 
patches with shapes ``(N, h, w, C)`` and ``(N, h, w, C')``, and to carry out 
any necessary padding, type conversion, overlapping and selection of channels.

This is accomplished through a series of custom `scikit-learn`_ transformers 
assembled into pipelines. These act on and return dask arrays.

Since RGB images and masks are treated a little differently (with respect to 
e.g. data normalisation) there are separate but similar pipelines which act 
simultaneously on the raw raster data and the ground truth masks.

Ultimately one such preprocessing pipeline will be allocated to each image 
or mask in a given dataset, converting each source file into a sequence of 
model-ready patches. Concatenation, shuffling, etc are carried 
out downstream by the :doc:`training` and :doc:`inference` interfaces.

Training pipelines
------------------

An instance of the default training pipeline for images can be easily 
constructed with the function
:func:`~gim_cv.preprocessing.get_image_training_pipeline`::
    
    from gim_cv.preprocessing import get_image_training_pipeline

    # default pipeline with patch size 256 * 256 and a fixed random seed
    X_pipeline = get_image_training_pipeline(window_dims=(256, 256), seed=42)
    X_pipeline

    Pipeline(steps = [
        ('channel_selector', ChannelSelector(channels=[0,1,2])),
        ('padder', ArrayPadder(window_dims=(256, 256))),
        ('resampler', ImageResampler(sf=1.0,
                                     preserve_int=True,
                                     resample_tolerance=1e-3)),
        ('tiler', OverlappingTiler(window_dims=(256, 256))),
        ('synchronised_shuffler', SynchronisedShuffler(seed=42))
    ])

This will use the first three (usually RGB) channels of the input raster array, 
pad this with zeros so that it divides evenly into multiples of the patch size, 
perform no resampling (in this case), and divide up the large array into non-overlapping
patches of size 256 * 256 before finally shuffling these. See the detailed API 
documentation for :mod:`gim_cv.preprocessing` for details on the individual transformers.

Note that normalisation (or whitening) is not applied as a preprocessing step, and is 
instead delegated to the batch generator functions (explained in the following sections).
This is because it can be useful to save the preprocessed arrays used for segmentation to 
speed up training (so they don't have to be generated every epoch from the source files),
and 8-bit integers representing RGB values are 4 x smaller than the 32-bit floats they will 
be converted to at training time.

An analogous default pipeline for binary masks can be constructed with the function 
:func:`~gim_cv.preprocessing.get_binary_mask_training_pipeline`::

    from gim_cv.preprocessing import get_binary_mask_training_pipeline

    y_pipeline = get_binary_mask_training_pipeline(window_dims=(256, 256), seed=42)
    y_pipeline

    Pipeline(steps = [
        ('dimension_adder', DimensionAdder()),
        ('padder', ArrayPadder(window_dims=(256, 256), constant_values=0)),
        ('resampler', ImageResampler(sf=1.0,
                                     preserve_int=True,
                                     is_mask=True,
                                     resample_tolerance=1e-3)),
        ('tiler', OverlappingTiler(window_dims=(256, 256))),
        ('synchronised_shuffler', SynchronisedShuffler(seed=42))
    ])
    
This performs largely the same tasks, with the exception of adding a length-one channel 
dimension to a binary mask so that multiclass and binary segmentation have data in the same 
form.

Inference pipelines
-------------------

Default pipelines for inference are similar, with the following caveats:

*   Since the mask quality of semantic segmentation models is known to decline 
    near the edges and corners of each patch, it makes sense to use as large a 
    patch as possible during inference and a batch size of 1. The maximum size 
    is limited by the VRAM available in the GPU for feature maps and will differ 
    from machine to machine.
*   Shuffling is not necessary or desirable for inference since the spatial 
    ordering of patches should be preserved.
*   Normalisation and conversion into floats is performed as part of the preprocessing 
    pipeline since there is no batch generator function during inference (each patch is
    exactly one batch). There is also no need to cache the arrays on disk since inference 
    takes one pass of a model over the input raster rather than many (as when training).

A default inference pipeline is accessible via :func:`~gim_cv.preprocessing.get_image_inference_pipeline`::


    from gim_cv.preprocessing import get_image_inference_pipeline

    X_pipeline = get_image_inference_pipeline(inference_window_size=896)
    X_pipeline

    Pipeline(steps=[
        ('channel_selector', ChannelSelector(channels=[0,1,2])),
        ('array_padder', ArrayPadder(896, 896)),
        ('resampler', ImageResampler(sf=1.0,
                                     preserve_int=True,
                                     resample_tolerance=1e-3)),
        ('tiler', Tiler(window_dims=(896, 896))),
        ('rechunker', Rechunker((1, -1, -1, -1))),
        ('scaler', SimpleInputScaler(1/255.)),
        ('float32er', Float32er())
    ])

Note the :class:`gim_cv.preprocessing.Rechunker` ensures that every dask chunk in 
the array contains one patch, so a dask chunk is equivalent to a batch which is 
convenient for just mapping a model over the chunks of the input dask array.

Custom pipelines for specific datasets
--------------------------------------

In certain cases the default pipelines won't be exactly suitable for a given dataset.
This can happen for example if mask labels are encoded as rasters with specific RGB 
values corresponding to different classes, and an additional step is required to 
encode the labels as one-hot vectors or binary values.

In this case one can override the default pipelines either completely or by 
prepending additional steps to the default ones.

When datasets defined in the :mod:`~gim_cv.datasets` module are used to construct 
:class:`~gim_cv.training.TrainingDataset` or :class:`~gim_cv.inference.InferenceDataset` 
objects (see :doc:`training` and :doc:`inference`), these delegate to the functions 
:func:`~gim_cv.datasets.get_image_training_pipeline_by_tag`, 
:func:`~gim_cv.datasets.get_binary_mask_training_pipeline_by_tag` and 
:func:`~gim_cv.datasets.get_image_inference_pipeline_by_tag`. By default these functions 
return the default pipelines outlined in the previous section. One can add dataset-specific 
behaviour by overriding the logic of these functions.

All use-cases so far have fallen into the category where the additional logic is confined to 
additional initial steps (after the extraction of the raster image/mask arrays). For convenience 
there exist global dictionaries in :mod:`gim_cv.datasets` module which allow one to inject 
additional ``sklearn`` Transformers based on the dataset tag. These will automatically be 
looked in by the pipe getter functions and prepended.

For example, the dictionary :obj:`gim_cv.datasets.BINARY_MASK_PIPELINE_PREP_STEPS` can be 
used as follows::

    # in datasets.py...
    from pathlib import Path
    import gim_cv.config as cfg
    from gim_cv.preprocessing import BinariserRGB

    # define the massachusetts buildings dataset
    massach_buildings_path = cfg.training_data_path / Path('mass_buildings')
    ds_massa = Dataset(
        tag='massachusetts',
        spatial_resolution=0.06,
        image_paths = sorted([d for d in massach_buildings_path.glob('train/sat/*')]),
        mask_paths = sorted([d for d in massach_buildings_path.glob('train/map/*')])
    )

    # specify that the mask arrays should first pass through an RGB binariser
    # (since buildings are encoded as red pixels in the source rasters)
    BINARY_MASK_PIPELINE_PREP_STEPS['massachusetts'] = [
        ('binariser', BinariserRGB((255, 0, 0)))
    ]

Now the mask pipeline associated with the dataset with the tag ``massachusetts`` will 
automatically have this additional step.

Image augmentation transformers
-------------------------------

Image augmentations are implemented using the `albumentations`_ library. The function 
:func:`~gim_cv.preprocessing.strong_aug` returns an albumentations ``Transformer`` which 
applies a set of image augmentations to batches with configurable probabilities. See the 
function's API documentation for more details.



.. _original AlexNet paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
.. _albumentations: https://github.com/albumentations-team/albumentations

.. _scikit-learn: https://scikit-learn.org/stable/data_transforms.html

