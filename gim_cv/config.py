"""
config.py

Loads a configuration yaml file 'config.yml' in this directory, executes
logic on its content and makes its contents available to other modules
by importing this module as e.g.:

    >> import gim_cv.config as cfg
    >> print(cfg.data_path)

`config.yml` contains various paths which allow one to define which directories
are used for storing data, models and results.
"""
import os
import yaml
from pathlib import Path

import logging


log = logging.getLogger(__name__)


# TODO: put this in an environment variable
def get_config_path(configpath_lmo='/home/root/config.yml',
                    configpath_lnn='/home/root/lnn/config.yml'):
    if os.path.exists(configpath_lmo):
        return configpath_lmo
    elif os.path.exists(configpath_lnn):
        return configpath_lnn
    else:
        msg = f"Files {configpath_lmo} or {configpath_lnn} not found!"
        log.error(msg)
        raise FileNotFoundError(msg)


def join(loader, node):
    """
    Function to handle joining paths in yaml file.

    When encountering '!join' tags, will treat subsequent items as
    a list of strings to be concatenated.

    Allows self-referencing paths like !join [\*BASE_PATH, /subdirectory/]
    """
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


# specify config yaml path default
YML_PATH = get_config_path()

def verify_path_exists(path, raise_exc=True):
    """ Check a path exists and raise a FileNotFoundError if not
    """
    if not path.exists():
        msg = f"File {path} not found!"
        if raise_exc:
            raise FileNotFoundError(msg)
        # warn
        log.warning(msg)

# -----------------------------------------------------------------------------
# register the tag handler
yaml.SafeLoader.add_constructor(tag='!join', constructor=join)

# check file exists
log.debug("Loading configuration file...")
yml_path = Path(YML_PATH)
verify_path_exists(yml_path)

# load the yml file as a dict
with open(yml_path, 'r') as f:
    try:
        _cfg = yaml.safe_load(f)
    except Exception as e:
        log.error(e)
        raise

# extra step - try to catch missing filepaths early
# make sure paths correspond to existing files, warning otherwise
for k, v in _cfg.items():
    if 'path' in k:
        _cfg[k] = Path(v)
        verify_path_exists(_cfg[k], raise_exc=False)

# update the local namespace with the values in the config dict, enabling
# us to access them when importing this file as attributes `config.model` etc
locals().update(_cfg)
