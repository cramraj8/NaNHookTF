# -*- coding: utf-8 -*-
# @author='Ramraj'

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging


class _NanTensorHookCuston(tf.train.SessionRunHook):
    """Monitors the loss tensor and stops training if loss is NaN.
    Can either fail with exception or just stop training.
    """

    def __init__(self, loss_tensor, fail_on_nan_loss=True):
        """Initializes a `NanTensorHook`.
        Args:
          loss_tensor: `Tensor`, the loss tensor.
          fail_on_nan_loss: `bool`, whether to raise exception when loss is NaN.
        """
        self._loss_tensor = loss_tensor
        self._fail_on_nan_loss = fail_on_nan_loss

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        if (np.isnan(run_values.results) or np.isinf(run_values.results)):
            failure_message = "Hey, what I am saying is !!$! Model diverged with loss = NaN."
            if self._fail_on_nan_loss:
                logging.error(failure_message)
                print('execution failed on NaN values.')
                raise IamThrowingNaNLossException
            else:
                logging.warning(failure_message)
                # We don't raise an error but we request stop without an exception.
                print('execution has NaN values.')
                run_context.request_stop()


def main(input_array):

    x = tf.placeholder(tf.float32)
    y = tf.multiply(x, 1, name='multiply')

    z = tf.reduce_sum(y, name='sum')

    hooks = [_NanTensorHookCuston(z, fail_on_nan_loss=True)]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:

        output = sess.run(z, feed_dict={x: input_array})


if __name__ == '__main__':

    x = np.array([[3.2, 1.2, 5.4], [3.3, np.NaN, np.NaN], [22.2, 4.3, 5.4]])

    main(x)
