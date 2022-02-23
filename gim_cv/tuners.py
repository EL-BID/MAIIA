"""
tuners.py

Provides custom kerastuner.Tuners for hyperparameter scans during model training.
"""
import collections
import copy

import numpy as np
import kerastuner as kt

import gim_cv.losses as losses
import gim_cv.tools.keras_one_cycle_clr as clr

from tensorflow import keras
from kerastuner.tuners import Hyperband
from kerastuner.engine import tuner_utils
import timbermafia as tm
import logging

log = logging.getLogger(__name__)

MOMENTUM_RANGE = (0.95, 0.85)


class HyperbandOCP(Hyperband, tm.Logged):
    """
    Hyperband tuner with one-cycle policy learning-rate-adjusting callback

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    attr1 : str
        Description of `attr1`.
    attr2 : :obj:`int`, optional
        Description of `attr2`.

    """
    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """
        Example function with PEP 484 type annotations.

        The return type must be duplicated in the docstring to comply
        with the NumPy docstring style.

        Parameters
        ----------
        trial : int
            The first parameter.
        *fit_args
            The second parameter.
        **fit_kwargs

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        hp = trial.hyperparameters
        if 'tuner/epochs' in hp.values:
            fit_kwargs['epochs'] = hp.values['tuner/epochs']
            fit_kwargs['initial_epoch'] = hp.values['tuner/initial_epoch']

        # create model checkpoint
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(
                trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True)
        original_callbacks = fit_kwargs.pop('callbacks', [])

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(fit_kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial.trial_id, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            # now add a one-cycle policy callback
            lr_min = hp.values['lr_min']
            lr_max = lr_min * hp.values['lr_multiplier']
            ocp = clr.OneCycle(lr_range=(lr_min, lr_max),
                               momentum_range=MOMENTUM_RANGE,
                               reset_on_train_begin=True,
                               record_frq=10)
            callbacks.append(ocp)
            copied_fit_kwargs['callbacks'] = callbacks

            model = self.hypermodel.build(trial.hyperparameters)
            history = model.fit(*fit_args, **copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step)
