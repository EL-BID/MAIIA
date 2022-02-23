Training
========

The :mod:`~gim_cv.training` module is responsible for providing classes which 
integrate datasets, preprocessing pipelines and batch/augmentation generators 
into a simple object which can be passed directly to a ``tf.keras`` model's ``fit`` 
method.

The main classes playing this role are :class:`~gim_cv.training.TrainingDataset` and 
:class:`~gim_cv.training.CompositeTrainingDataset`. These can be used standalaone 
(by specifying which image/mask files to create from directly) or be created directly 
from a :class:`~gim_cv.datasets.Dataset` object (comprised of RGB images and ground 
truth masks). We will cover both of these cases in turn.

For the very short version, look at :ref:`load_training_data`.

Training Datasets
-----------------

A :class:`~gim_cv.training.TrainingDataset` converts *one pair* of corresponding 
image and mask files into a set of preprocessed training/validation/testing patch
arrays, and accepts the following main arguments:

* Paths to image and mask source files (ideally in ``.tif`` and/or ``.shp`` format)
* A function which, when called, creates a preprocessing pipeline for the images
* A function which, when called, creates a preprocessing pipeline for the masks
* A batch generator function (usually you will want :func:`~gim_cv.training.fancy_batch_generator`) which 
  accepts dask arrays of associated image and mask patches (the output of the preprocessing 
  pipelines) and yields from these batches of (augmented) images.

There are also various options for controlling training/validation/testing splits 
and pruning functions to eliminate invalid arrays (such as empty arrays with all 
white or all black pixels). See the :class:`~gim_cv.training.TrainingDataset` 
API documentation for more details.

Once the :class:`~gim_cv.training.TrainingDataset` has been created, its 
:meth:`~gim_cv.training.TrainingDataset.prepare` method must be called (this will 
create an instance of the necessary preprocessing pipelines, then create the Dask 
task graph corresponding to loading the arrays from the source files and performing 
all of the preprocessing operations). Once the prepare method has been called the 
image and mask patch dask arrays will be accessible as the attributes ``X`` and ``y``.

After this stage, the training dataset object will have access to the generator methods 
:meth:`~gim_cv.training.TrainingDataset.batch_gen_train`, 
:meth:`~gim_cv.training.TrainingDataset.batch_gen_val` and (optionally) 
:meth:`~gim_cv.training.TrainingDataset.batch_gen_test` which feed directly to your model.

Here's an example::

    from functools import partial
    from gim_cv.preprocessing import (
        get_image_training_pipeline, get_binary_mask_training_pipeline, # default pipelines
        get_aug_datagen # basic augs
    )
    from gim_cv.training import TrainingDataset, pair_batch_generator
    
    
    # get function to generate batches of images and masks for each dataset,
    # choosing a batch size and augmentation generator
    batch_generator = partial(pair_batch_generator, 
                              batch_size=batch_size,
                              img_aug=get_aug_datagen(), # simple augs (v/h flip) for brevity
                              mask_aug=get_aug_datagen(), # simple augs (v/h flip) for brevity
                              seed=seed,
                              shuffle=False)
    
    tds = TrainingDataset(
        './training_data/image_1.tif',
        './training_data/mask_1.tif',
        image_pipeline_factory=get_image_training_pipeline,
        mask_pipeline_factory=get_binary_mask_training_pipeline,
        batch_generator_fn=batch_generator,
    )
    
     # build preprocessing operations
     tds.prepare() # now arrays are generated as the X, y attributes
    
    # here you can train a model which accepts (B, h, w, C/C') image/mask arrays
    model.fit(tds.batch_gen_train(),
              validation_data=training_data.batch_gen_val(),
              ...)

Composite Training Datasets
---------------------------

Most of the time you will want to train a model on multiple source image and mask 
files, from a :class:`~gim_cv.datasets.Dataset` comprised of multiple files and/or 
from multiple :class:`~gim_cv.datasets.Dataset` objects (for example, of different 
areas and/or the same areas over multiple years).

The :class:`~gim_cv.training.CompositeTrainingDataset` class is a wrapper for a set 
of individual :class:`~gim_cv.training.TrainingDataset` objects which includes logic 
to consolidate all the individual arrays of image/mask patches together and shuffle 
these, to prune the final large patch arrays (eliminating empty ones to speed up 
training) and to cache the shuffled and preprocessed arrays in ``.zarr`` format on 
disk to greatly speed up training.

A composite training dataset can be created by just adding training datasets to 
each other::

    tds1 = TrainingDataset(
        './training_data/image_1.tif',
        './training_data/mask_1.tif',
        image_pipeline_factory=get_image_training_pipeline,
        mask_pipeline_factory=get_binary_mask_training_pipeline,
        batch_generator_fn=batch_generator,
    )

    tds2 = TrainingDataset(
        './training_data/image_2.tif',
        './training_data/mask_2.tif',
        image_pipeline_factory=get_image_training_pipeline, # these may be different to above!
        mask_pipeline_factory=get_binary_mask_training_pipeline,
        batch_generator_fn=batch_generator,
    )

    # create a CompositeTrainingDataset by adding together as many TrainingDatasets as you like
    tds = tds1 + tds2

    # create full arrays
    tds.prepare()

A list of the constituent :class:`~gim_cv.training.TrainingDataset` objects are then available 
via the ``constituents`` attribute.

Like training datasets, composite training datasets have a 
:meth:`~gim_cv.training.CompositeTrainingDataset.prepare` method which is responsible for 
first delegating to the constituents' ``prepare`` methods (to queue up loading and 
preprocessing patches from each pair of files) and then performing concatenation, shuffling 
and chunk optimisation to produce the combined image and mask arrays. 

Caching/loading/deleting of the combined arrays in ``.zarr`` format is performed through the 
methods :meth:`~gim_cv.training.CompositeTrainingDataset.save_prepared_arrays`, 
:meth:`~gim_cv.training.CompositeTrainingDataset.load_prepared_arrays` and 
:meth:`~gim_cv.training.CompositeTrainingDataset.delete_prepared_arrays` methods. The directory 
in which these cached arrays are stored can be fixed by first setting the 
:attr:`~gim_cv.training.CompositeTrainingDataset.cache_directory` attribute. The dask arrays 
for ``X`` and ``y`` will now point to the cached preprocessed version which is much faster 
to read from::

    # set the zarr cache directory
    tds.cache_directory = cfg.proc_data_path / Path("my_combined_dataset")
    # save the arrays
    tds.save_prepared_arrays() # tds.X, tds.y will now point to the cache on disk

Once arrays are cached for a given set of training datasets, you can just load them directly::

    # next time
    tds = tds1 + tds2
    # skip prepare, just load directly
    tds.load_prepared_arrays()

You can combine the save/load logic in a script by doing something like::

    tds.cache_directory = cfg.proc_data_path / Path("my_combined_dataset")
    try:
        tds.load_prepared_arrays()
        log.info(f"Using training data arrays cached at: {tds.cache_directory}")
    except (ValueError, FileNotFoundError, KeyError) as v:
        log.warning("No cached training arrays found. Generating them...")
        tds.prepare()
        tds.save_prepared_arrays()

A composite training dataset must be assigned a batch generator function (like an 
individual training dataset) to produce batches from the monolithic image/mask arrays. 
Here's an example of using the :func:`~gim_cv.preprocessing.strong_aug` function 
(Albumentations augmentations) with :class:`~gim_cv.preprocessing.FancyPCA` enabled::

    from functools import partial
    from dask.distributed import Client
    from gim_cv.training import fancy_batch_generator
    from gim_cv.preprocessing import FancyPCA, strong_aug

    # create dataset and prepare/load full arrays from cache...

    # create dask client to parallelise augmentations
    client = Client(processes=False)
    # calculate FancyPCA colour axes from the data
    fpca = FancyPCA(tds.X_train, alpha_std=.3, p=1.0)
    # create the augmentation transformer
    augger = strong_aug(p=.8, fancy_pca=fpca)
    # assign a batch generator using these
    tds.batch_generator_fn = partial(
        fancy_batch_generator,
        batch_size=batch_size,
        augger=augger,
        client=client,
        seed=seed,
        shuffle_blocks_every_epoch=True,
        shuffle_within_blocks=True,
        float32=True
    )

Now a model can be trained in the usual way::

    model.fit(tds.batch_gen_train(),
              validation_data=training_data.batch_gen_val(),
              ...)

.. _load_training_data:

Creating training dataset objects from predefined datasets
----------------------------------------------------------

You can create a :class:`~gim_cv.training.CompositeTrainingDataset` directly from a 
:class:`gim_cv.datasets.Dataset` object using the :meth:`gim_cv.datasets.Dataset.load_training_data` 
method. This will construct a :class:`~gim_cv.training.TrainingDataset` from each image 
/ mask pair defined in the :class:`gim_cv.datasets.Dataset`'s ``image_paths`` and ``mask_paths`` 
attributes, combine these and return a :class:`~gim_cv.training.CompositeTrainingDataset`::

    from gim_cv.datasets import get_dataset
    
    # select some datasets
    ds_inr = get_dataset('phil_man_14_50cm')
    ds_pot = get_dataset('viet_hcm_20_50cm')

    # create a CompositeTrainingDataset from all of the image/mask files present for each
    tds_in = ds_in.load_training_data(
        batch_size=4,
        train_val_test_split=(0.9, 0.1),
    )
    tds_pot = ds_pot.load_training_data(
        batch_size=4,
        train_val_test_split=(0.9, 0.1),
    )
    # combine these
    tds_all = tds_in + tds_pot
    # build image and mask dask arrays tds_all.X and tds_all.y...
    tds_all.prepare() 
    # set batch generator function, train model etc...

The :meth:`gim_cv.datasets.Dataset.load_training_data` method will use the default pipelines for 
each dataset (see :doc:`preprocessing` documentation) by default but has arguments which 
allow these to be overridden. See the detailed API documentation for more details.

Training script
---------------

For a fully-fledged training script using the ``Segmentalist`` model with 
datasets predefined in the :mod:`gim_cv.datasets` module, you can check out the 
documentation, comments and help string for ``bin/train_segmentalist.py``::

    $ python train_segmentalist.py --help

This allows one to select any set of datasets defined in the datasets module, and use 
these to train a ``Segmentalist`` model with configurable hyperparameters and architectural 
features. It also saves the model weights and associated run parameters to disk for later 
analysis and comparison.

The script also has an option for Stochastic Weight Averaging (see 
`Averaging Weights Leads to Wider Optima and Better Generalization (Izmailov et al. 2018)`_).

For example, to train a model on the datasets ``manilla2014`` and ``ho-chi-minh2020`` resampled (if appropriate) 
to 0.3m spatial resolution, and with spatial attention gates, deep supervision and pyramid pooling 
enabled, with overlapping patches and the tversky loss function for 80 epochs, in the container 
environment in the ``bin`` directory run::

    $ python train_segmentalist.py -d phil_man_14_50cm,viet_hcm_20_50cm -tsr 0.5 -sag -ds -pp -ot -l tversky_loss -ep 80

The script will create a unique directory containing a ``.yml`` file with all the parameters 
passed to the script (so all model hyperparameters, datasets etc). As the model completes 
epochs of training, the weights will also be stored in this directory for future use. These 
directories are created by default within:: 

    cfg.models_path / Path('ebs_trained_models')

and can be read programmatically to compare results between different runs (see the next section).

.. _comparing trained models:

Comparing trained models
------------------------

The codebase contains utilities for parsing the directories created by the training script 
described in the previous section and loading the weights based on checkpoint loss values or 
any other criteria.

Since this is tied to a particular model and the implementation of the existing training 
script, it is not part of the library code and thus lives next to the training script 
in ``gim-cv/bin/utils.py``.

Selecting a good model by some criteria is obviously required when running inference. 
The function :func:`bin.utils.collate_run_data` builds a pandas dataframe containing all 
the best model checkpoints created by each run of the training script along with the 
associated run parameters.

Here is an example of how to achieve that using the directories created by the current 
training script::

    import sys
    # point to the bin directory i.e. where "train_segmentalist.py" and the 
    # associated "utils.py" are
    sys.path.append('../../bin')

    import gim_cv.config as cfg
    from gim_cv.models.segmentalist import Segmentalist

    # load all the directories created by the training script in models_dir and put the run data into a pandas dataframe
    df_segm_trained = utils.collate_run_data(
        models_dir=cfg.models_volume_path, # this should match the "models_dir" arg used in train_segmentalist.py
        model_name='Segmentalist' # this will search for run data saved in directories beginning "Segmentalist"
    )

    # sort by validation loss and pick the best row for a given loss function and training data used
    # let's say weighted binary cross entropy and imagery of Manilla AOI 4 + Manilla AOI 5. 
    # you can also add other conditions to the query, for example 'deep_supervision == False'
    df_sorted = df_segm_trained.sort_values(by='lowest_val_loss').query(
        'datasets == "phil_man_14_50cm_04,phil_man_14_50cm_05" and loss_fn == "wbce_adaptive"'
    )
    best_row = df_sorted.iloc[0]

    # load the best model using the saved weights corresponding to the lowest validation loss
    model = Segmentalist.load_from_metadata(row=best_row)

    # you can load the validation or testing data used during training to get a feel for performance
    X_val = np.load(row.X0_val)
    # run inference on first batch of validation data to visualise results
    y_val_pred = model(X_val[:2]/255.)

    # now do inference, etc...



.. _Averaging Weights Leads to Wider Optima and Better Generalization (Izmailov et al. 2018): http://arxiv.org/abs/1803.05407
