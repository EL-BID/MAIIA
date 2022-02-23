import logging
import sys

import timbermafia as tm

import gim_cv.config as cfg

from pathlib import Path

tm.basic_config(
    palette='sensible',
    style='compact',
    stream=sys.stdout,
    level=cfg.log_level,
    name='gim-cv'
    #filename=str(cfg.logs_path / Path("output.log"))
)

#log = logging.getLogger(__name__)
#log.setLevel()
