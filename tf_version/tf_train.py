
r"""
My simplest raw TF code for training a network and saving ckpt files.
All written in raw TF; except model architecture, which is in tf/slim.
No visual plot codes.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/basic_session_run_hooks.py


https://stackoverflow.com/questions/42647403/tf-train-nantensorhookloss-fail-on-nan-loss-false-will-still-raise-exception

"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import time


class NanTensorHook_Ram_Created(tf.train.SessionRunHook):
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
                raise IamThrowingNaNLossException
            else:
                logging.warning(failure_message)
                # We don't raise an error but we request stop without an exception.
                run_context.request_stop()


def load_data_set(name=None):

    data_feed = pd.read_csv('Brain_Integ_X.csv', skiprows=[0], header=None)
    labels_feed = pd.read_csv('Brain_Integ_Y.csv', skiprows=[1], header=0)
    survival = labels_feed['Survival']
    censored = labels_feed['Censored']

    survival = survival.values
    censored = censored.values
    data = data_feed.values
    data = np.float32(data)

    censored_survival = survival[censored == 1]
    censored_data = data[censored == 1]

    y = np.asarray(censored_survival)
    x = np.asarray(censored_data)

    print('Shape of X : ', x.shape)
    print('Shape of Y : ', y.shape)
    return (x, y)


def multilayer_neural_network_model(inputs, HIDDEN_LAYERS, BETA,
                                    scope="deep_regression_model"):
    with tf.variable_scope(scope, 'deep_regression', [inputs]):
        end_points = {}
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(BETA)):
            net = slim.stack(inputs,
                             slim.fully_connected,
                             HIDDEN_LAYERS,
                             scope='fc')
            end_points['fc'] = net
            predictions = slim.fully_connected(net, 1, activation_fn=None,
                                               scope='prediction')
            end_points['out'] = predictions
            return predictions, end_points


LEARNING_RATE = 1
BETA = 0.001
TRAINING_EPOCHS = 300
BATCH_SIZE = 100
DISPLAY_STEP = 100
DROPOUT_RATE = 0.9
VARIANCE = 0.1

INITIAL_LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY_FACTOR = 0.7
NUM_OF_EPOCHS_BEFORE_DECAY = 1000

# ============ Network Parameters ============
HIDDEN_LAYERS = [500, 400, 600, 500, 500, 500]
N_CLASSES = 1
# ******************************************************************************
with tf.Graph().as_default() as graph:
    logging.set_verbosity(tf.logging.INFO)      # *****====
    data_x, data_y = load_data_set()
    data_x, data_y = shuffle(data_x, data_y, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                        test_size=0.20,
                                                        random_state=420)
    total_observations = x_train.shape[0]
    input_features = x_train.shape[1]
    ckpt_dir = './log/'
    if not tf.gfile.Exists(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)
    num_batches_per_epoch = total_observations / BATCH_SIZE
    num_steps_per_epoch = num_batches_per_epoch
    decay_steps = int(NUM_OF_EPOCHS_BEFORE_DECAY * num_steps_per_epoch)
    global_step = tf.Variable(0, name='global_step', trainable=False)   # *****====

    x = tf.placeholder("float", [None, input_features], name='features')
    y = tf.placeholder("float", [None], name='labels')

    pred, end_points = multilayer_neural_network_model(x, HIDDEN_LAYERS, BETA)

    lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

# ********************** LOSS && OPTIMIZE *************************************
    loss = tf.reduce_sum(tf.square(tf.transpose(pred) - y), name="loss") / (2 * total_observations)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)


# ******************************************************************************
    # Launch the graph
    hooks = [NanTensorHook_Ram_Created(loss, fail_on_nan_loss=True)]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        # tf.global_variables_initializer fails the MoniteredSession()
        # sess.run(tf.global_variables_initializer())

        # Saver fails the MoniteredSession()
        # saver = tf.train.Saver()

        for epoch in range(TRAINING_EPOCHS + 1):
            avg_cost = 0.0
            total_batch = int(total_observations / BATCH_SIZE)
            for i in range(total_batch - 1):    # Loop over all batches
                batch_x = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                batch_y = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                _, batch_cost, batch_pred = sess.run([optimizer,
                                                      loss,
                                                      pred],
                                                     feed_dict={x: batch_x,
                                                                y: batch_y})
                # Run optimization (backprop) and cost op (to get loss value)
                avg_cost += batch_cost / total_batch   # Compute average loss

            if epoch % DISPLAY_STEP == 0:         # Display logs per epoch step
                print("Epoch : ", "%05d" % (epoch + 1),
                      " cost = ", " {:.9f} ".format(avg_cost))
                for i in xrange(3):  # Comparing orig, predicted values
                    print("label value:", batch_y[i], "predicted value:",
                          batch_pred[i])
                print("------------------------------------------------------")
        print('Global_Step final value : ', sess.run(global_step))
        # Saving the Graph and Variables in Checkpoint files
        saver.save(sess, ckpt_dir + 'deep_regression_trained_model',
                   global_step=global_step)
        pred_out = sess.run(pred, feed_dict={x: x_train})

        print("Training Finished!")

        # TESTING PHASE #
        _, test_predicted_values = sess.run([loss, pred],
                                            feed_dict={x: x_test,
                                                       y: y_test})
        for i in range(len(y_test)):
            print("Labeled VALUE : ", y_test[i], " \t\t\t \
                >>>> Predicted VALUE : ", float(test_predicted_values[i]))
