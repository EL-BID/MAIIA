"""train_deep_resunet.py

Script to train a DeepResUNet model on a given selection of training datasets,
with various hyperparameters and options (spatial resolution to be resampled to, 
augmentations to use etc.) as script arguments.

The resulting model weights will be saved in a directory created by the script
along with information about the arguments the script was called with to allow 
comparison of different models. This directory will be named with a unique id
generated to distinguish it.

The default directory in which models are saved is the models_directory 
specified in config.yml / 'ebs_trained_models'. This is where we currently 
mount the large shared drive for this purpose (the mount point is set in 
docker-compose.yml).

Running with mostly the default arguments is recommended as these are tuned to give 
decent results. Arguments to use are:

-d (datasets)
-tsr (target spatial resolution)
-l (loss function)

For more details, run this script with --help to see the available options.
"""
import re
import operator
import logging
import pickle
import time
import argparse

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import dask
import dask.array as da
import tensorflow as tf
import kerastuner as kt
import sys
import pprint
import yaml
import uuid
import albumentations as A
import joblib 

import gim_cv.config as cfg
import gim_cv.utils as utils
import gim_cv.losses as losses
import gim_cv.datasets as datasets
import gim_cv.tools.keras_one_cycle_clr as clr

from functools import partial, reduce
from pathlib import Path
from time import perf_counter as pc

#from sklearn.externals import joblib
from tensorflow.keras.models import load_model
from distributed import Client, LocalCluster
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
from osgeo import gdal, ogr, osr
from zarr.errors import ArrayNotFoundError

from gim_cv.models.deepresunet import HyperDeepResUNet, DeepResUNet
from gim_cv.training import TrainingDataset, pair_batch_generator, CompositeTrainingDataset, fancy_batch_generator
from gim_cv.datasets import (TrainingDataLoader,
                             get_dataset,
                             get_image_training_pipeline_by_tag,
                             get_binary_mask_training_pipeline_by_tag,
                             list_datasets)
from gim_cv.preprocessing import get_aug_datagen, FancyPCA, strong_aug, balanced_oversample
from gim_cv.utils import plot_pair, parse_kwarg_str
from gim_cv.tuners import HyperbandOCP

# --- set up script command line arguments
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-s', '--patch-size', default=256, type=int, dest='patch_size',
                    help=('The patch size used for training - the model will see'
                          '(patch_size * patch_size) square sections of the training datasets'))
parser.add_argument('-ep', '--epochs', default=60, type=int, help="no. training epochs", dest='epochs')
parser.add_argument('-bs', '--batch_size', default=4, type=int, help="batch_size", dest='batch_size')
parser.add_argument('-p', '--patience', default=10, type=int,
                    help="number of epochs of stagnation of validation loss before early stopping of training",
                    dest='patience')
parser.add_argument('-md', '--models-dir', dest='models_dir',
                    default=str(cfg.models_path / Path('ebs_trained_models/')),
                    help='directory in which to store model checkpoints and metrics')
parser.add_argument('-rs', '--seed', dest='seed', type=int, default=cfg.seed,
                    help="random seed")
parser.add_argument('-vf', '--val-frac', dest='val_frac', type=float, default=0.1,
                    help="validation fraction - the fraction of data used for validation")
parser.add_argument('-tf', '--test-frac', dest='test_frac', type=float, default=0.,
                    help="test fraction - the fraction of data used for testing (sectioned off and dumped to disk)")
parser.add_argument('-lr', '--lr-init', dest='lr_init', type=float, default=0.00001,
                    help="initial learning rate")
parser.add_argument('-lrmin', '--lr-min', dest='lr_min', type=float, default=0.000001, 
                    help='minimum learning rate if reduce LR on plateau callback used')
parser.add_argument('-lrf', '--lr-reduce-factor', dest='lr_reduce_factor', type=float, default=0.5,
                    help='multiplicative LR reduction factor for reduce LR on plateau callback')
parser.add_argument('-lrp', '--lr-reduce-patience', dest='lr_reduce_patience', type=int, default=2,
                    help='epochs patience for LR reduction application if reduce LR on plateau')
parser.add_argument('-ocp', '--use-ocp', dest='ocp', action='store_true', default=False,
                    help="enable one-cycle policy (not used atm)")
parser.add_argument('-dt', '--dump-test-data', dest='dump_test_data', default=False, 
                    action='store_true', help='dump the test arrays to zarr')
parser.add_argument('-da', '--dump-first-batches', dest='dump_first_batches', default=True,
                    action='store_true',
                    help='precalculate first chunk of training array and dump to disk for inspection')
parser.add_argument('-l', '--loss-fn', dest='loss_fn', type=str, default='wbce_adaptive',
                    help='loss function name as string (looks in building_age.losses).'
                    'optionally provide kwargs afterwards using a colon to delineate the '
                    'beginning of comma-separated keyword args, e.g. '
                    'custom_loss_fn:gamma=1.5,alpha=0.2')
parser.add_argument('-vl', '--use-val', dest='use_val', default=True, action='store_true',
                    help='switch: evaluate on validation data every epoch and track this')
parser.add_argument('-c', '--use-cache', dest='use_cache', default=True, action='store_true',
                    help='try to read the preprocessed arrays from file if serialised')
parser.add_argument('-sc', '--save-to-cache', dest='save_to_cache', default=True, action='store_true',
                    help='save the preprocessed arrays to file for future training runs')
parser.add_argument('-opt', '--optimiser', dest='optimiser', default='adam', type=str,
                    help='gradient descent optimizer (adam or sgd)')
parser.add_argument('-tsr', '--target-spatial-res', dest='target_spatial_resolution', default=.2, type=float,
                    help='spatial resolution to resample to')
parser.add_argument('-fa', '--fancy-augs', dest='fancy_augs', default=True, action='store_true',
                    help='Flag whether to use fancy augmentations (albumentations + FancyPCA)')
parser.add_argument('-f', '--filters', dest='filters', default='64,128,256,512,1024', type=str, 
                    help='comma delimited string of integers for resblock filters')
parser.add_argument('-k', '--kernel-size', dest='kernel_size', default=7, type=int,
                    help='integer size of kernels in first conv layer')
parser.add_argument('-ba', '--balanced-oversample', dest='balanced_oversample', default=False, action='store_true',
                    help='oversample training arrays to balance different datasets. makes an "epoch" much longer.')
parser.add_argument('-d','--datasets',dest='datasets',type=str,default='lux_belair_19_true_20cm',
                    help=('Comma delimited string of dataset tags. Available datasets are:\n'
                        f'{datasets.list_datasets(skip_missing_files=True)}'))

args = parser.parse_args()

assert tf.test.is_gpu_available(), "CHECK GPU AVAILABILITY! (eg /etc/docker/daemon.json default runtime)"

log = logging.getLogger(__name__)

if __name__ == '__main__':
    
    # seed RNG
    np.random.seed(args.seed)

    # set window size
    patch_dims = (args.patch_size, args.patch_size)
    # determine the partitioning of the training data into train, validation and test from args
    if args.test_frac:
        train_val_test_split = (1 - args.val_frac - args.test_frac, args.val_frac, args.test_frac)
    else:
        train_val_test_split = (1.-args.val_frac, args.val_frac)

    # --- assemble training datasets
    # get dataset tags - remove extraneous spaces from names and sort so that order
    # is not important
    dataset_tags = sorted([d.lstrip(' ').rstrip(' ') for d in args.datasets.split(',')])
    
    #  build each training dataset individually
    tdsets = []
    for ds_tag in dataset_tags:
        ds = datasets.get_dataset(ds_tag)
        # TODO: refactor to make load_training_data one step.
        training_loader = datasets.TrainingDataLoader(
            batch_size=args.batch_size,
            train_val_test_split=train_val_test_split,
            seed=args.seed
        )
        tdsets.append(
            ds.load_training_data(
                loader=training_loader,
                window_size=args.patch_size,
                resample_factor=ds.spatial_resolution/args.target_spatial_resolution)
        )

    # combine all training datasets into one big training dataset
    if len(tdsets) == 1:
        tds = tdsets[0]
    else:
        tds = reduce(operator.add, tdsets)
                    
    # optionally supply an oversampling function to ensure equal representation
    if args.balanced_oversample:
        tds.oversample_fn = balanced_oversample
                    
    # create a string to identify the combination of datasets and the spatial resolution
    # this can be used to fix a directory to cache arrays for future use from the same data
    data_res_str = f"data_{tds.tags_str}_target_res_{args.target_spatial_resolution}"
    # set the cache directory to save preprocessed arrays in an appropriately named directory
    tds.cache_directory = cfg.proc_data_path / Path(data_res_str)

    # --- preprocess training data
    # generate arrays from rasters on-the-fly at training time rather than using cache
    if not args.use_cache:
        tds.prepare()
    # look for cached arrays if they're already there. this speeds up training considerably.
    else:
        log.info(f"Searching for cached training data at {tds.cache_directory}...")
        try:
            tds.load_prepared_arrays()
            log.info(f"Using training data arrays cached at: {tds.cache_directory}")
        except (ValueError, FileNotFoundError, ArrayNotFoundError, KeyError) as v:
            log.warning("No cached training arrays found. Generating them...")
            tds.prepare()
            log.info("Generating arrays:")
            log.info(tds.X)
            log.info(tds.y)
            # save the generated arrays to the cache directory to speed up future runs
            if args.save_to_cache:
                log.info(f"Saving processed training data to {tds.cache_directory}...")
                t0 = pc()
                try:
                    tds.save_prepared_arrays()
                except KeyboardInterrupt:
                    tds.delete_prepared_arrays()
                    sys.exit(1)
                log.warning(f"Arrays stored! took {pc()-t0:.2f}s!")
                tds.load_prepared_arrays()
                log.warning(f"Using training data arrays cached at: {tds.cache_directory}")               
    
    # --- assign data generator for scaling, augmentations etc
    # default: fancy augmentations using Albumentations transforms
    if args.fancy_augs:
        log.warning("~~ Fancy augs on ~~")
        # start distributed cluster for mapping augmentations
        client = Client(processes=False)
        log.info("Calculating PCA decomposition of training RGBs...")
        # perform "fancy PCA", calculating data-driven RGB shifts on training data
        fpca = FancyPCA(tds.X_train, alpha_std=.3, p=1.0, max_samples=1000)
        log.warning(f"Eigenvalues are {fpca.sampler.eig_vals}")
        log.warning(f"Eigenvectors are {fpca.sampler.eig_vecs}")
        # create the augmentation generator, including fancy PCA
        augger = strong_aug(p=.8, fancy_pca=fpca)
        # now create the batch generator function with augmentations
        tds.batch_generator_fn = partial(
            fancy_batch_generator,
            batch_size=args.batch_size,
            augger=augger,
            client=client,
            seed=args.seed,
            shuffle_blocks_every_epoch=True,
            shuffle_within_blocks=True,
            float32=True
        )
        aug_sfx = 'fancy'
    # basic augmentations: horizontal and vertical flips
    else:
        tds.batch_generator_fn = partial(
            pair_batch_generator,
            batch_size=args.batch_size,
            img_aug=get_aug_datagen(horizontal_flip=True,
                                    vertical_flip=True),
            mask_aug=get_aug_datagen(horizontal_flip=True, #args....
                                    vertical_flip=True),
            seed=args.seed,
            #shuffle=True
        )
        aug_sfx = 'basic'
        
    # --- configure model training
    # get loss function and any kwargs (entered as a string to script arguments)
    loss_fn_name, *lf_kwarg_str = args.loss_fn.split(':')
    # grab func itself from losses module by name
    loss_fn = getattr(losses, loss_fn_name)
    # optionally provide kwargs to higher-order function to return lf (needed if your 
    # loss function is parametrised and so takes arguments other than y_true, y_pred)
    # (this should return a tf.keras style 2-parameter fn with signature y_true, y_pred)
    if lf_kwarg_str:
        lf_kwargs = parse_kwarg_str(*lf_kwarg_str)
        loss_fn = loss_fn(**lf_kwargs)
    else:
        lf_kwargs = {}
    # encode loss function args as a cleaned-up string for identifying models trained with this
    lfastr = '_args_' + '_'.join([f'{k}={v:.2f}' for k, v in lf_kwargs.items()]) if lf_kwargs else ''                             
                    
    # calculate number of training steps (batches per epoch)
    train_steps = tds.X_train.shape[0]//args.batch_size
    valid_steps = tds.X_val.shape[0]//args.batch_size
    # check validation data isn't empty
    if args.use_val:
        assert tds.X_val.shape[0] > 0, (
            "Not enough dask blocks to make up validation data frac!\n"
            f"train: {tds.X_train}"
        )
        
    # select metrics - standard segmentation metrics
    metrics = [
        losses.tversky_index,
        losses.jaccard_index,
        losses.recall,
        losses.precision,
        losses.specificity,
        losses.npv,
        losses.dice_coefficient
    ]

    # interpret optimizer
    # classic SGD with momentum
    if args.optimiser == 'sgd':
        opt = tf.keras.optimizers.SGD(
            learning_rate=args.lr_init, momentum=0.85, nesterov=False
        )
    # ADAM
    elif args.optimiser == 'adam':
        opt = tf.keras.optimizers.Adam(
            learning_rate=args.lr_init, beta_1=0.9, beta_2=0.999, amsgrad=False
        ) # check out RADAM?
    else:
        raise ValueError(f"Optimiser {opt} not understood")        
    
    # specify training directory to save weights and metrics for this loss_fn and data ID
    # within models_dir
    # generates a unique id (uuid4) per training run
    project_name = Path(f'DeepResUNet_{uuid.uuid4()}')
    training_dir = Path(args.models_dir) / project_name
    training_dir.mkdir(parents=True, exist_ok=True)
                    
    # -- set up training callbacks
    # early stopping - terminate if validation loss stagnates for patience epochs
    monitor = 'val_loss'
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor, patience=args.patience)
    ]

    # reduce the learning rate on loss plateaus
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor,
                                             factor=args.lr_reduce_factor,
                                             patience=args.lr_reduce_patience,
                                             min_lr=args.lr_min)
    )

    # set up tensorboard to record metrics in a subdirectory
    tb_pth = training_dir / Path("metrics/")
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=str(tb_pth),
        update_freq=50
    )
    callbacks.append(tb_cb)

    # set up weight checkpoints in the training directory
    cp_fmt = 'cp-e{epoch:02d}-ji{jaccard_index:.5f}-l{loss:.5f}'
    suffix = '-vl{val_loss:.5f}.ckpt'
    cp_fmt = cp_fmt + suffix
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(training_dir / Path(cp_fmt)), # saved_model
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    cp_callback_trn = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(training_dir / Path(cp_fmt)), # saved_model
        monitor='loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    callbacks.append(cp_callback)
    callbacks.append(cp_callback_trn)

    # --- build and compile the model
    model = DeepResUNet(initial_conv_kernel=(args.kernel_size, args.kernel_size),
                        filters=[int(f) for f in args.filters.split(',')])
    model.build(input_shape=(None, None, None, 3))
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=metrics)
    model.predict(np.random.rand(args.batch_size, args.patch_size, args.patch_size, 3))

    # option to save first batches of training data as a standalone reference to validate
    # that the training data looks as expected
    if args.dump_first_batches:
        log.warning(f"Dumping first batches to {training_dir}...")
        # training arrays
        np.save(f'{training_dir}/X_train_{data_res_str}.npy', tds.X_train.blocks[0].compute())
        np.save(f'{training_dir}/y_train_{data_res_str}.npy', tds.y_train.blocks[0].compute())
        # validation arrays
        np.save(f'{training_dir}/X_val_{data_res_str}.npy', tds.X_val.compute())
        np.save(f'{training_dir}/y_val_{data_res_str}.npy', tds.y_val.compute())

    # option to dump test data - arrays partitioned off and not seen or used to validate
    # useful for independent and unbiased check of model performance
    if args.dump_test_data:
        if args.test_frac:
            log.warning(f"Dumping testing arrays to {training_dir}...")
            try:
                tds.X_test.to_zarr(f'{training_dir}/X_test.zarr')
                tds.y_test.to_zarr(f'{training_dir}/y_test.zarr')
            except ValueError:
                # already present most likely
                pass
                    
    # dump arguments to this script and augmentations to file for cross-checking
    # future runs
    # augmentations
    if args.fancy_augs:
        A.save(augger, f'{training_dir}/transform.yml', data_format='yaml')
    # args to this script
    with open(f'{training_dir}/run_params.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    log.info("Start training...")
    # --- train the model
    model.fit(
        tds.batch_gen_train(),
        steps_per_epoch=train_steps, # steps per epoch
        epochs=args.epochs,
        validation_data=tds.batch_gen_val(),
        validation_steps=valid_steps,
        max_queue_size=50,
        callbacks=callbacks
    )
