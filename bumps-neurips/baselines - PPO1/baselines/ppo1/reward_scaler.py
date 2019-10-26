from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf

class RewardScaler(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
    def _init(self):
        rw = U.get_placeholder(name="rw", dtype=tf.float32, shape=[])

        with tf.variable_scope("rwfilter"):
            self.rw_rms = RunningMeanStd(shape=[])
        
        rwz = tf.clip_by_value((rw - self.rw_rms.mean) / self.rw_rms.std, -10.0, 10.0)

        self._scale = U.function([rw], [rwz, self.rw_rms.mean, self.rw_rms.std])

    def scale(self, rw):
        return self._scale(rw)