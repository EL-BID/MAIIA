""" interfaces/write.py
"""
import shutil
import abc
import dask.array as da
import timbermafia as tm

from pathlib import Path
from typing import Union, Any

import logging

log = logging.getLogger(__name__)



class BaseMaskWriter(tm.Logged):
    pass

class InferenceDataset(tm.Logged):
    def __init__(self, untile_method='simple'):
        output_filename +=
        self.untiler = get_untiler(self.untile_method)
    def run_inference(self, model):
        tiled_input_array = input_pipeline.fit_transform(img.arr)
        tiled_output_array = model.predict(tiled_input_array)
        self.untiler.fit_transform(tiled_output_array)

# pseudocode for solution
tiler = SimpleTiler()

class Infertiler(BaseEstimator, tm.Logged):
    def __init__(self, model, tiler):
        self.model = model
        self.tiler = tiler
    def fit(self, X, y=None):
        self.tiled_input_ = tiler.fit_transform(X)
        return self
    def transform(self, X, y=None):
        # will have to make blocks here
        self.tiled_output_ = self.model.predict(self.tiled_input_)
        output = self.tiler.inverse_transform()



untiler = SimpleUntiler()
#untiler = HannUntiler()
untiler.fit_transform(tiled_array)
