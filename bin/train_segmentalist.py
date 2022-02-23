"""
Trains a Segmentalist model on a (set of) dataset(s) and saves the resulting model checkpoints.

Accepts various hyperparameters and configurables - see the --help option for a description of these.

Uses datasets defined in gim_cv.datasets.
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
import tensorflow_addons as tfa
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
from tensorflow_addons.optimizers import SWA
#from zarr.errors import ArrayNotFoundError

from gim_cv.models.segmentalist import Segmentalist
from gim_cv.training import TrainingDataset, pair_batch_generator, CompositeTrainingDataset, fancy_batch_generator
from gim_cv.datasets import (get_dataset,
                             get_image_training_pipeline_by_tag,
                             get_binary_mask_training_pipeline_by_tag,
                             list_datasets)
from gim_cv.preprocessing import get_aug_datagen, FancyPCA, strong_aug, balanced_oversample
from gim_cv.utils import plot_pair, parse_kwarg_str
from gim_cv.tuners import HyperbandOCP

log = logging.getLogger()
#assert len(log.handlers) == 1
#handler = log.handlers[0]
#handler.setLevel(logging.INFO)

# --- set up script command line arguments
parser = argparse.ArgumentParser()
# training dataset selection
parser.add_argument('-d', '--datasets', dest='datasets', type=str,
                    default='train_tif',
                    help='Comma delimited string of dataset tags. Available datasets are:\n'
                         f'{datasets.list_datasets(skip_missing_files=True)}')
parser.add_argument('-tsr', '--target-spatial-res', dest='target_spatial_resolution', default=0.4, type=float,
                    help='spatial resolution to resample to. native resolution for all datasets if == 0.')
# model features
parser.add_argument('-pp', '--input-pyramid', dest='pyramid_pooling', action='store_true', help='Enable input pyramid')
parser.add_argument('-npp', '--no-input-pyramid', dest='pyramid_pooling', action='store_false', help='Disable input pyramid')
parser.set_defaults(pyramid_pooling=False)
parser.add_argument('-ds', '--deep-supervision', dest='deep_supervision', action='store_true', help='Enable deep supervision')
parser.add_argument('-nds', '--no-deep-supervision', dest='deep_supervision', action='store_false', help='Disable deep supervision')
parser.set_defaults(deep_supervision=False)
parser.add_argument('-lc', '--lambda-conv', dest='lambda_conv', action='store_true',
                    help='Replace main 3x3 convolutions in residual blocks with Lambda convolutions')
parser.add_argument('-nlc', '--no-lambda-conv', dest='lambda_conv', action='store_false',
                    help='Don\'t replace main 3x3 convolutions in residual blocks with Lambda convolutions')
parser.set_defaults(lambda_conv=False)
parser.add_argument('-ecbam', '--encoder-cbam', dest='encoder_cbam', action='store_true',
                    help='enable CBAM blocks in encoder residual blocks')
parser.add_argument('-necbam', '--no-encoder-cbam', dest='encoder_cbam', action='store_false',
                    help='disable CBAM blocks in encoder residual blocks')
parser.add_argument('-dcbam', '--decoder-cbam', dest='decoder_cbam', action='store_true',
                    help='enable CBAM blocks in decoder residual blocks')
parser.add_argument('-ndcbam', '--no-decoder-cbam', dest='decoder_cbam', action='store_false',
                    help='disable CBAM blocks in decoder residual blocks')
parser.set_defaults(encoder_cbam=False)
parser.set_defaults(decoder_cbam=False)
# mutually model features: attention gate type in decoder stages (if any)
group = parser.add_mutually_exclusive_group()
group.add_argument('-sag', '--attention-gate', dest='sag', action='store_true',
                   help='Enable spatial attention gate')
group.add_argument('-nsag', '--no-attention-gate', dest='sag', action='store_false',
                   help='Disable spatial attention gate')
group.set_defaults(sag=False)
group.add_argument('-csag', '--channel-spatial-attention-gate', dest='csag', action='store_true', default=True,
                   help='Enable channel-spatial attention gate')
group.add_argument('-ncsag', '--no-channel-spatial-attention-gate', dest='csag', action='store_true', default=True,
                   help='Disable channel-spatial attention gate')
group.set_defaults(csag=False)
# model hyperparameters
parser.add_argument('-lb', '--layer-blocks', dest='layer_blocks', type=str, default='2,2,2,2',
                   help=(
                       'Comma-delimited list of the number of residual blocks per layer of the encoder. The last number fixes '
                       'those in the bridge which is unique. The decoder mirrors these blocks excluding the bridge. The final '
                       'block of the decoder uses the last_layer_decoder_blocks argument to fix the number of residual convblocks.'
                   ))
parser.add_argument('-ldb', '--last-decoder-layer-blocks', dest='last_decoder_layer_blocks', type=int, default=2,
                   help='The number of residual conv blocks in the final decoder block.')
parser.add_argument('-if', '--initial-filters', dest='initial_filters', type=int, default=64,
                   help='The number of filters in the first large-kernel-size ResNet convolution in the encoder.')
parser.add_argument('-rf', '--residual-filters', dest='residual_filters', type=str, default='128,256,512,1024',
                   help=('Comma-delimited list of the number of filters in the encoder blocks. The last fixes '
                         'the number of filters in the bridge block. The decoder blocks mirror these excluding the bridge. '
                         'The final decoder block uses the same number of filters as the initial_filters argument.'))
parser.add_argument('-ik', '--initial-kernel-size', dest='initial_kernel_size', type=int, default=7,
                   help='The kernel size for the initial convolution in the encoder block. Usually ResNet style 7x7.')
parser.add_argument('-hk', '--head-kernel-size', dest='head_kernel_size', type=int, default=1,
                   help='The kernel size for the head (final segmentation layer). Typically 1x1.')
parser.add_argument('-cd', '--cardinality', dest='cardinality', type=int, default=1,
                   help='The cardinality of ResNeXt grouped convolutions in main blocks (if lambda_conv is false).')
#parser.add_argument('-ce', '--channel-expansion-factor', dest='channel_expansion_factor', type=int, default=2,
#                   help='The factor by which the number of channels is expanded at the end of each encoder block.')
parser.add_argument('-act', '--activation', dest='activation', type=str, default='relu',
                   help='String name of activation function used throughout.')
parser.add_argument('-dsmp', '--downsample', dest='downsample', type=str, default='pool',
                   help='Mechanism used for downsampling feature maps: "pool" or "strides".')
# training hyperparameters                   
parser.add_argument('-s', '--patch-size', default=256, type=int, dest='patch_size')
parser.add_argument('-ot', '--overlap-tiles', dest='overlapping_tiles', action='store_true', 
                    help='Flag to toggle on overlapping tiles in training data (half-step)')
parser.add_argument('-not', '--no-overlap-tiles', dest='overlapping_tiles', action='store_false',  
                    help='Flag to toggle off overlapping tiles in training data (half-step) (no overlapping).')
parser.set_defaults(overlapping_tiles=False)
parser.add_argument('-ep', '--epochs', default=100, type=int, help="no. training epochs", dest='epochs')
parser.add_argument('-bs', '--batch_size', default=4, type=int, help="batch_size", dest='batch_size')
parser.add_argument('-l', '--loss-fn', dest='loss_fn', type=str, default='wbce_adaptive',
                    help='loss function name as string (looks in building_age.losses).'
                    'optionally provide kwargs afterwards using a colon to delineate the '
                    'beginning of comma-separated keyword args, e.g. '
                    'custom_loss_fn:gamma=1.5,alpha=0.2')
parser.add_argument('-opt', '--optimiser', dest='optimiser', default='adam', type=str,
                    help='gradient descent optimizer (adam, sgd or ranger)')
parser.add_argument('-swa', '--stochastic-weight-averaging', dest='swa', action='store_true',
                    help='apply stochastic weight averaging to optimizer')
parser.add_argument('-nswa', '--no-stochastic-weight-averaging', dest='swa', action='store_false',
                    help='do not apply stochastic weight averaging to optimizer')
parser.set_defaults(swa=True)
parser.add_argument('-dswa', '--duration-swa', dest='duration_swa', default=50, type=int, 
                    help='number of epochs before last where SWA is applied')
parser.add_argument('-pswa', '--period-swa', dest='period_swa', default=5, type=int,
                    help='period in epochs over which to average weights with SWA')
parser.add_argument('-vl', '--use-val', dest='use_val', default=True, action='store_true',
                    help='switch: evaluate on validation data every epoch and track this')
parser.add_argument('-p', '--patience', default=7, type=int, help="patience", dest='patience')
parser.add_argument('-rs', '--seed', dest='seed', type=int, default=cfg.seed,
                    help="random seed")
parser.add_argument('-vf', '--val-frac', dest='val_frac', type=float, default=0.1,
                    help="validation fraction")
parser.add_argument('-tf', '--test-frac', dest='test_frac', type=float, default=0.,
                    help="test fraction")
parser.add_argument('-fa', '--fancy-augs', dest='fancy_augs', default=True, action='store_true',
                    help='Flag whether to use fancy augmentations (albumentations + FancyPCA)')
parser.add_argument('-lr', '--lr-init', dest='lr_init', type=float, default=0.0001,
                    help="initial learning rate")
parser.add_argument('-lrmin', '--lr-min', dest='lr_min', type=float, default=0.000001, 
                    help='minimum learning rate if reduce LR on plateau callback used')
parser.add_argument('-lrf', '--lr-reduce-factor', dest='lr_reduce_factor', type=float, default=0.5,
                    help='multiplicative LR reduction factor for reduce LR on plateau callback')
parser.add_argument('-lrp', '--lr-reduce-patience', dest='lr_reduce_patience', type=int, default=2,
                    help='epochs patience for LR reduction application if reduce LR on plateau')
parser.add_argument('-ocp', '--use-ocp', dest='ocp', action='store_true', default=False,
                    help="enable one-cycle policy (not used atm)")
parser.add_argument('-ba', '--balanced-oversample', dest='balanced_oversample', default=False, action='store_true',
                    help='oversample training arrays to balance different datasets. makes an "epoch" much longer.')
# weight/parameter/array-saving-specific parameters                    
parser.add_argument('-md', '--models-dir', dest='models_dir',
                    default=str(cfg.models_path), # str(cfg.models_path / Path('ebs_trained_models')),
                    help='directory in which to store model checkpoints and metrics')
parser.add_argument('-dt', '--dump-test-data', dest='dump_test_data', default=False, 
                    action='store_true', help='dump the test arrays to zarr')
parser.add_argument('-da', '--dump-first-batches', dest='dump_first_batches', default=True,
                    action='store_true',
                    help='precalculate first chunk of training array and dump to disk for inspection')
parser.add_argument('-c', '--use-cache', dest='use_cache', default=True, action='store_true',
                    help='try to read the preprocessed arrays from file if serialised')
parser.add_argument('-sc', '--save-to-cache', dest='save_to_cache', default=True, action='store_true',
                    help='save the preprocessed arrays to file for future training runs')


args = parser.parse_args()

# sort datasets so order not important
args.datasets = ','.join(sorted(args.datasets.split(',')))

if __name__ == '__main__':

    assert tf.test.is_gpu_available(), "CHECK GPU AVAILABILITY! (eg /etc/docker/daemon.json default runtime)"

    np.random.seed(args.seed)

    # set window/patch size
    patch_dims = (args.patch_size, args.patch_size)
    # process necessary parser arguments
    args.layer_blocks_ = [int(n) for n in args.layer_blocks.split(',')]
    args.residual_filters_ = [int(n) for n in args.residual_filters.split(',')]
    args.initial_kernel_size_ = (args.initial_kernel_size, args.initial_kernel_size)
    args.head_kernel_size_ = (args.head_kernel_size, args.head_kernel_size)
    
    # decide partitioning into train/validation/test
    if args.test_frac:
        train_val_test_split = (1 - args.val_frac - args.test_frac, args.val_frac, args.test_frac)
    else:
        train_val_test_split = (1.-args.val_frac, args.val_frac)

    # --- assemble training datasets
    # get dataset tags - sort to fix order to identify different permutations for array caching
    dataset_tags = sorted([d.lstrip(' ').rstrip(' ') for d in args.datasets.split(',')])
    
    # get each of the training datasets requested
    tdsets = []
    for ds_tag in dataset_tags:
        ds = datasets.get_dataset(ds_tag)
        rf = ds.spatial_resolution/args.target_spatial_resolution if args.target_spatial_resolution else 1.
        tdsets.append(
            ds.load_training_data(
                batch_size=args.batch_size,
                train_val_test_split=train_val_test_split,
                seed=args.seed,
                window_size=args.patch_size,
                overlap_tiles=args.overlapping_tiles,
                resample_factor=rf
            )
        )

    # combine them into one big (composite) training dataset
    if len(tdsets) == 1:
        tds = tdsets[0]
    else:
        tds = reduce(operator.add, tdsets)
    # optionally oversample to equally represent (not recommended)
    if args.balanced_oversample:
        tds.oversample_fn = balanced_oversample
        
    # create a string to identify the combination of datasets and the spatial resolution
    # used in saving model checkpoints to quickly identify training data used
    data_res_str = f"data_{tds.tags_str}_target_res_{args.target_spatial_resolution}"
    if args.overlapping_tiles:
        data_res_str += '_overlapping_tiles'
    # set the cache directory to save preprocessed arrays in an appropriately named directory
    tds.cache_directory = cfg.proc_data_path / Path(data_res_str)

    # --- preprocess training data
    # generate arrays from rasters on-the-fly at training time
    if not args.use_cache:
        tds.prepare()
    # look for cached arrays if they're already there. this speeds up training considerably.
    else:
        log.info(f"Searching for cached training data at {tds.cache_directory}...")
        try:
            tds.load_prepared_arrays()
            log.info(f"Using training data arrays cached at: {tds.cache_directory}")
        except (ValueError, FileNotFoundError, KeyError) as v:
            log.warning("No cached training arrays found. Generating them...")
            tds.prepare()
            log.info("Generating arrays:")
            log.info(tds.X)
            log.info(tds.y)
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
    # albumentations + fancyPCA => "fancy" augs. recommended.
    if args.fancy_augs:
        log.warning("~~ Fancy augs on ~~")
        # start distributed cluster for mapping augmentations
        client = Client(processes=False)
        log.info("Calculating PCA decomposition of training RGBs...")
        fpca = FancyPCA(tds.X_train, alpha_std=.3, p=1.0)
        log.warning(f"Eigenvalues are {fpca.sampler.eig_vals}")
        log.warning(f"Eigenvectors are {fpca.sampler.eig_vecs}")
        augger = strong_aug(p=.8, fancy_pca=fpca)
        tds.batch_generator_fn = partial(
            fancy_batch_generator,
            batch_size=args.batch_size,
            augger=augger,
            client=client,
            seed=args.seed,
            shuffle_blocks_every_epoch=True,
            shuffle_within_blocks=True,
            deep_supervision=args.deep_supervision, #False
            float32=True
        )
        aug_sfx = 'fancy'
    # basic batch generator - flip augs only. not recommended.
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
    # get loss function and any kwargs (entered as a string to argparser)
    loss_fn_name, *lf_kwarg_str = args.loss_fn.split(':')
    # grab func itself from losses module by name
    loss_fn = getattr(losses, loss_fn_name)
    # optionally provide kwargs to higher-order function to return lf
    # (this should return a tf.keras style 2-parameter fn with signature y_true, y_pred)
    if lf_kwarg_str:
        lf_kwargs = parse_kwarg_str(*lf_kwarg_str)
        loss_fn = loss_fn(**lf_kwargs)
    else:
        lf_kwargs = {}
    # encode loss function args as a cleaned-up string for identifying models trained with this
    lfastr = '_args_' + '_'.join([f'{k}={v:.2f}' for k, v in lf_kwargs.items()]) if lf_kwargs else ''                             
                    
    # calculate number of training and validation steps
    train_steps = tds.X_train.shape[0]//args.batch_size
    valid_steps = tds.X_val.shape[0]//args.batch_size
    print(f'train_steps={train_steps}')
    print(f'val_steps={valid_steps}')
    if args.use_val:
        assert tds.X_val.shape[0] > 0, (
            "Not enough dask blocks to make up validation data frac!\n"
            f"train: {tds.X_train}"
        )
        
    # select metrics
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
    if args.optimiser == 'sgd':
        opt = tf.keras.optimizers.SGD(
            learning_rate=args.lr_init, momentum=0.85, nesterov=False
        )
    elif args.optimiser == 'adam':
        opt = tf.keras.optimizers.Adam(
            learning_rate=args.lr_init, beta_1=0.9, beta_2=0.999, amsgrad=False
        ) # check out RADAM?
    elif args.optimiser == 'ranger':
        radam = tfa.optimizers.RectifiedAdam(lr=args.lr_init, min_lr=args.lr_min)
        opt = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    else:
        raise ValueError(f"Optimiser {opt} not understood")
    # stoachastic weight averaging if enabled
    if args.swa:
        opt = SWA(opt, start_averaging=args.epochs - args.duration_swa, average_period=args.period_swa)
    
    # specify training directory to save weights and metrics for this loss_fn and data ID
    # within models_dir
    project_name = Path(f'Segmentalist_{uuid.uuid4()}')
    training_dir = Path(args.models_dir) / project_name
    training_dir.mkdir(parents=True, exist_ok=True)
    # -- callbacks

    # early stopping
    monitor = 'val_loss'
    callbacks= [] #[tf.keras.callbacks.EarlyStopping(monitor, patience=args.patience)]

    # reduce the learning rate on plateaus
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

    # set up checkpoints in the training directory
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
    # format attention gate param
    if args.csag:
        ag = 'CSAG'
    elif args.sag:
        ag = 'SAG'
    else:
        ag = None
    model = Segmentalist(
        n_classes=tds.y_train.shape[-1],
        layer_blocks=args.layer_blocks_,
        last_decoder_layer_blocks=args.last_decoder_layer_blocks,
        initial_filters=args.initial_filters,
        residual_filters=args.residual_filters_,
        initial_kernel_size=args.initial_kernel_size_,
        head_kernel_size=args.head_kernel_size_,
        cardinality=args.cardinality,
        act=args.activation,
        downsample=args.downsample,
        decoder_attention_gates=ag,
        encoder_cbam=args.encoder_cbam,
        decoder_cbam=args.decoder_cbam,
        pyramid_pooling=args.pyramid_pooling,
        deep_supervision=args.deep_supervision,
        lambda_conv=args.lambda_conv,
    )
    model.build(input_shape=(args.batch_size, args.patch_size, args.patch_size, tds.X_train.shape[-1]))
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=metrics)

    # option to save first arrays for quick check of consistency
    if args.dump_first_batches:
        log.warning(f"Dumping first batches to {training_dir}...")
        #training_dir.mkdir(parents=True, exist_ok=True)
        np.save(f'{training_dir}/X_train_{data_res_str}.npy', tds.X_train.blocks[0].compute())
        np.save(f'{training_dir}/y_train_{data_res_str}.npy', tds.y_train.blocks[0].compute())
        np.save(f'{training_dir}/X_val_{data_res_str}.npy', tds.X_val.compute())
        np.save(f'{training_dir}/y_val_{data_res_str}.npy', tds.y_val.compute())
        #sys.exit(0)
    # option to dump test data
    if args.dump_test_data:
        if args.test_frac:
            log.warning(f"Dumping testing arrays to {training_dir}...")
            try:
                tds.X_test.to_zarr(f'{training_dir}/X_test.zarr')
                tds.y_test.to_zarr(f'{training_dir}/y_test.zarr')
            except ValueError:
                # already present most likely
                pass
    # dump setup
    # augmentations
    if args.fancy_augs:
        A.save(augger, f'{training_dir}/transform.yml', data_format='yaml')
    # dump args passed to this script to a yaml file for comparison of runs later
    with open(f'{training_dir}/run_params.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    log.info("Start training...")
    # --- train the model
    print('fit model')
    model.fit(
        tds.batch_gen_train(),
        steps_per_epoch=train_steps, # steps per epoch
        epochs=args.epochs,
        validation_data=tds.batch_gen_val(),
        validation_steps=valid_steps,
        max_queue_size=50,
        callbacks=callbacks
    )
    log.info("Training ended successfully ...")
