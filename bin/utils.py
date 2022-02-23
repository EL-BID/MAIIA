import logging
import yaml

import dask.array as da
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import tensorflow as tf
import cufflinks as cf
import plotly.offline as pyo
import plotly.graph_objs as go

import gim_cv.config as cfg
import gim_cv.datasets as datasets
import gim_cv.losses as losses

from functools import partial
from pathlib import Path

from gim_cv.preprocessing import get_aug_datagen
from gim_cv.training import pair_batch_generator
from gim_cv.models.deepresunet import DeepResUNet
from gim_cv.models.attention_unet import attn_reg



# Propagate notebook log level to called modules
logger = logging.getLogger()
#assert len(logger.handlers) == 1
#handler = logger.handlers[0]
#handler.setLevel(logging.ERROR)

def join(loader, node):
    """
    Function to handle joining paths in yaml file.

    When encountering '!join' tags, will treat subsequent items as
    a list of strings to be concatenated.

    Allows self-referencing paths like !join [\*BASE_PATH, /subdirectory/]
    """
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class LoaderWithTuples(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

LoaderWithTuples.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    LoaderWithTuples.construct_python_tuple)

def get_run_data(model_dir,
                 cp_ptn=(
                     'cp-e(?P<epoch>\d+)(-ji(?P<ji>[\d.]+))?-l(?P<loss>[\d.]+)'
                     '-vl(?P<val_loss>[\d.]+).ckpt.index'
                 ),
                 dataset_aliases=None):
    """
    Parse a model directory name, possibly containing checkpoints, to 
    reconstruct the loss function, training data used and best loss achieved
    
    Returns
    -------
    :obj:`pd.DataFrame`
        A dataframe containing training metadata such as loss, dataset used, spatial 
        resolution and the paths to the best checkpoints
    """
    # get uuid4
    mtch = re.match('.*_(?P<uuid>(\w+-){,4}\w+)$',str(model_dir.parts[-1]))
    uuid = mtch.group('uuid')
    # find checkpoints
    ckpts = model_dir.glob('*.ckpt.index')
    base_paths, losses, val_losses, epochs = [], [], [], []
    for ckpt in ckpts:
        base_paths.append(str(ckpt)[:-6]) # remove index
        cp_match = re.match(
            cp_ptn,
            str(ckpt.parts[-1])
        )
        epochs.append(cp_match.group('epoch'))
        losses.append(cp_match.group('loss'))
        val_losses.append(cp_match.group('val_loss'))
    best_trn_loss_ix = np.argmin(losses) if losses else 0
    best_val_loss_ix = np.argmin(val_losses) if val_losses else 0
    try:
        low_l_cp = base_paths[best_trn_loss_ix]
        low_l = losses[best_trn_loss_ix]
    except IndexError as e:
        low_l_cp = None
        low_l = None
    try:
        low_vl_cp = base_paths[best_val_loss_ix]
        low_vl = val_losses[best_val_loss_ix]
    except IndexError as e:
        low_vl_cp = None
        low_vl = None
    try:
        X0_train = list(model_dir.glob('X_train*.npy'))[0]
        y0_train = list(model_dir.glob('y_train*.npy'))[0]
    except IndexError:
        X0_train = None
        y0_train = None
    try:
        X0_val = list(model_dir.glob('X_val*.npy'))[0]
        y0_val = list(model_dir.glob('y_val*.npy'))[0]
    except IndexError:
        X0_val = None
        y0_val = None
    metadata = {
        'uuid4':uuid,
        'training_dir' : str(model_dir),
        'lowest_loss_ckpt' : low_l_cp,
        'lowest_val_loss_ckpt' : low_vl_cp,
        'lowest_loss' : np.float32(low_l),
        'lowest_val_loss' : np.float32(low_vl),
        'X0_train' : X0_train,
        'y0_train' : y0_train,
        'X0_val' : X0_val,
        'y0_val' : y0_val
    }
    with open(model_dir / Path('run_params.yml')) as f:
        params = yaml.load(f, Loader=LoaderWithTuples)
    metadata.update(params)
    if dataset_aliases:
        metadata['datasets_alias'] = dataset_aliases.get(metadata['datasets'], metadata['datasets'])
    return pd.Series(metadata)


def collate_run_data(models_dir, model_name='Segmentalist', dataset_aliases=None):
    """
    Creates a dataframe containing metadata on all trained models in a directory
    
    model_name can be a regex
    """
    runs = []
    for m in models_dir.glob(f'{model_name}_*'):
        if re.match(f'{model_name}''_(\w+-){,4}\w+$',str(m.parts[-1])):
            try:
                runs.append(get_run_data(m, dataset_aliases=dataset_aliases))
            except Exception as e:
                print(f"{m} failed with exception:")
                print(e)
    if not runs:
        raise Exception(f"No model directories found at {models_dir}!")
    return pd.concat(runs, axis=1).T


def load_sample_arrays(row, val=True):
    """
    Loads and returns the sample X, y arrays (first block of ~500 image/mask pairs)
    
    If val is false, load the training arrays
    """
    if val:
        if row.X0_val:
            return (np.load(row.X0_val), np.load(row.y0_val))
        return None, None
    else:
        if row.X0_train:
            return (np.load(row.X0_train), np.load(row.y0_train))
        return None, None


def load_best_model(row,
                    val=True,
                    **model_kwargs):
    """
    Loads a DeepResUNet with the weights loaded from the checkpoint with 
    the lowest validation loss.
    
    If val is False, uses the checkpoint with the lowest training loss.
    """
    initial_conv_kernel = (int(row.kernel_size), int(row.kernel_size))
    filters = row.filters.split(',')
    model = DeepResUNet(
        initial_conv_kernel=initial_conv_kernel,
        filters=[int(f) for f in filters],
        **model_kwargs)
    # use same setup as training script to ensure parity in loaded model
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
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
    # TODO - implement loss function args if necessary
    loss_fn = getattr(losses, row.loss_fn)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=metrics)
    cp = row.lowest_val_loss_ckpt if val else row.lowest_loss_ckpt
    if cp:
        model.load_weights(cp)
        return model
    else:
        raise ValueError("No checkpoint found in directory!")
        
        
def load_best_attention_unet_model(row, val=True, **model_kwargs):
    # --- build and compile the model
    model = attn_reg(input_size=(row.patch_size, row.patch_size, 3))
    lf = getattr(losses, row.loss_fn)
    loss_ = {'pred1': lf,
            'pred2': lf,
            'pred3': lf}
    if 'focal_tversky' in row.loss_fn:
        loss_['final'] = losses.tversky_loss
    else:
        loss_['final'] = lf
    
    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}
    # use same setup as training script to ensure parity in loaded model
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
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
    model.compile(optimizer=opt,
                  loss=loss_,
                  loss_weights=loss_weights,
                  metrics=metrics)
    cp = row.lowest_val_loss_ckpt if val else row.lowest_loss_ckpt
    if cp:
        model.load_weights(cp)
        return model
    else:
        raise ValueError("No checkpoint found in directory!")  
        
        
def get_metric_df(X, y, model, verbose=0):
    i = 0
    results = []
    for i in range(X.shape[0]):
        results.append(
            model.evaluate(x=X[np.newaxis,i]/255., y=y[np.newaxis, i]/1., steps=1, verbose=verbose)
        )
        if i % 50 == 0:
            logger.debug(f"{i}/{y.shape[0]}")
    return pd.DataFrame(
        columns=model.metrics_names,
        data=results
    )

def select_lowest_losses_by_dataset(df, group_by=['datasets', 'loss_fn', 'fancy_augs']):
    idx = df.groupby(group_by)['lowest_val_loss'].transform(min) == df['lowest_val_loss']
    return df[idx]


from collections import namedtuple

class Results:
    def __init__(self,
                 keys=('loss_fn', 'training_datasets', 'fancy_augs', 'inference_datasets', 'train_or_val'),
                 results={}):
        self.key_gen = namedtuple(f'{self.__class__.__name__}_result_key', keys) 
        self.results = dict()     
    def __iter__(self):
        items = self.results.items()
        yield from items
    def add(self, result,  **kwargs):
        key = self.key_gen(**{k:v for k,v in kwargs.items() if k not in ('self', 'result')})
        #print('*')
        #print(f"add result with {self.key_gen} and key {key} of type {type(result)}")
        self.results[key] = result
        #print(f"i am a {self.__class__.__name__} and my results are now")
        #print(self.results)
        #print("*")
    def get(self,  **kwargs):
        key = self.key_gen(**{k:v for k,v in kwargs.items() if k!='self'})
        return self.results[key]
    
    
class Arrays(Results):
    def add_from_metadata(self, metadata):
        datasets = metadata.datasets_alias
        fancy_augs = metadata.fancy_augs
        X_train, y_train = load_sample_arrays(metadata, val=False)
        X_val, y_val = load_sample_arrays(metadata, val=True)
        self.add((X_train, y_train), datasets=datasets, train_or_val='train', fancy_augs=fancy_augs)
        self.add((X_val, y_val), datasets=datasets, train_or_val='val', fancy_augs=fancy_augs)
        

class Model:
    def __init__(self, metadata, network=None):
        self.metadata = metadata
        self.network = network
    def load(self, fn=load_best_model):
        self.network = fn(self.metadata)
    def unload(self):
        self.network = None  
    
    
class Models:
    key_gen = namedtuple('model_key', ('loss_fn', 'training_datasets', 'fancy_augs'))
    def __init__(self, models={}):
        self.models = models
    def __iter__(self):
        items = self.models.items()
        yield from items
        
    def add(self, model):
        key = self.key_gen(
            loss_fn=model.metadata.loss_fn,
            training_datasets=model.metadata.datasets_alias,
            fancy_augs=model.metadata.fancy_augs
        )
        self.models[key] = model
    def get(self, *, loss_fn, training_datasets, fancy_augs):
        key = self.key_gen(loss_fn=loss_fn, training_datasets=training_datasets, fancy_augs=fancy_augs)
        return self.models[key]
    def add_from_metadata(self, row):
        self.add(Model(row))

        
class Metrics(Results):
    def generate_with_model(self, X, y, model, datasets, train_or_val):
        result = get_metric_df(X, y, model.network)
        self.add(
            result,
            loss_fn=model.metadata.loss_fn,
            training_datasets=model.metadata.datasets_alias,
            inference_datasets=datasets,
            train_or_val=train_or_val,
            fancy_augs=model.metadata.fancy_augs
        )

class InferenceResults(Results):
    pass
