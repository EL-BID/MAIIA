import logging

import timbermafia as tm

from tensorflow.keras.callbacks import Callback

log = logging.getLogger(__name__)


class NBatchLogger(Callback, tm.Logged):
    """
    A Logger that log average performance per `display` steps.

    initial credit:
    https://github.com/keras-team/keras/issues/2850#issuecomment-371353851
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            self.log.debug('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()
