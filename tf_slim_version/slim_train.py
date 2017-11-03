# -*- coding: utf-8 -*-
# @author='Ramraj'

r"""There are 2 ways to overcome NaN effect.

1. tf.check_numerics() in slim.learning.create_train_op() will check for
    NaN or inf and throws exception.
    So we can handle the exception to do some desired work.

2. If we set check_numerics=False in slim.learning.create_train_op()
    Loss may have NaN.
    So we need to seek for NaNTensorHook.


"""


from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from SurvivalAnalysis import SurvivalAnalysis
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('x_data', './data/Brain_Integ_X.csv',
                    'Directory file with features-data.')
flags.DEFINE_string('y_data', './data/Brain_Integ_Y.csv',
                    'Directory file with label-values.')
flags.DEFINE_string('ckpt_dir', './ckpt_dir/',
                    'Directory for checkpoint files.')
flags.DEFINE_float('split_ratio', 0.6,
                   'Split ratio for test data.')
flags.DEFINE_float('lr_decay_rate', 0.9,
                   'Learning decaying rate.')
flags.DEFINE_float('beta', 0.01,
                   'Regularizing constant.')
flags.DEFINE_float('dropout', 0.6,
                   'Drop out ratio.')
flags.DEFINE_float('init_lr', 0.001,
                   'Initial learning rate.')
flags.DEFINE_integer('batch_size', 100,
                     'Batch size.')
flags.DEFINE_integer('n_epochs', 500,
                     'Number of epochs.')
flags.DEFINE_integer('n_classes', 1,
                     'Number of classes in case of classification.')
flags.DEFINE_integer('display_step', 100,
                     'Displaying step at training.')
flags.DEFINE_integer('n_layers', 3,
                     'Number of layers.')
flags.DEFINE_integer('n_neurons', 500,
                     'Number of Neurons.')


FLAGS = flags.FLAGS
slim = tf.contrib.slim


def data_providers(x_data_file='./data/Brain_Integ_X.csv',
                   y_data_file='./data/Brain_Integ_Y.csv'):
    """
    This function reads the data file and extracts the features and labelled
    values.
    Then according to that patient is dead, only those observations would be
    taken care
    for deep learning trainig.

    Args:
        data_file: list of strings representing the paths of input files. Here the input features and ground truth values are separated in 2 files.
    Returns:
        `Numpy array`, extracted feature columns and label column.
    Example:
        >>> read_dataset()
        ( [[2.3, 2.4, 6.5],[2.3, 5.4,3.3]], [12, 82] )
    """

    data_feed = pd.read_csv(x_data_file, skiprows=[0], header=None)
    labels_feed = pd.read_csv(y_data_file, skiprows=[1], header=0)
    survival = labels_feed['Survival']
    censored = labels_feed['Censored']

    survival = survival.values
    censored = censored.values
    data = data_feed.values
    data = np.float32(data)

    censored_survival = survival[censored == 1]
    censored_features = data[censored == 1]
    censored_data = censored[censored == 1]

    y = np.asarray(censored_survival, dtype=np.int32)
    x = np.asarray(censored_features)
    c = np.asarray(censored_data, dtype=np.int32)

    print('Shape of X : ', x.shape)
    print('Shape of Y : ', y.shape)
    print('Shape of C : ', c.shape)
    return (x, y, c)


def multilayer_nn_model(inputs, hidden_layers, n_classes, beta,
                        scope="deep_regression"):
    """Creates a deep regression model.

    This function takes input as the required parameters to build a deep
    neural network and builds the layer-wise network. Once its instance is
    called by parsing the input feedings, this function will perform the
    feed-forward and returns the output layer responses.

    Args:
        inputs: A node that yields a `Tensor` of size [total_observations,
        input_features].

    Returns:
        predictions: `Tensor` of shape (1) (scalar) of response.
        end_points: A dict of end points representing the hidden layers.
    """

    end_points = {}
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(beta)):
        net = slim.stack(inputs,
                         slim.fully_connected,
                         hidden_layers,
                         scope='fc')
        end_points['fc'] = net
        predictions = slim.fully_connected(net, n_classes, activation_fn=None,
                                           scope='prediction')
        end_points['out'] = predictions
        return predictions, end_points


def estimate_cost(prediction, at_risk_label, observed):
    """The function for calculating the loss interms of log partial likelihood.

    This function brings output vector from the deep neural network and the
    calculated survival risk vector. Then it calculates the log partial
    likelihood only for the observed patients.

    Args
    ----
        prediction : numpy foat32 array representing the output of DNN.
        at_risk_label : numpy folat32 representing the at_risk score.
        observed : numpy int32 representing non-censored patient status.

    Returns
    -------
        cost : numpy float32 scalar representing the cost calculated from the
        prediction by DNN.

    """

    n_observations = at_risk_label.shape[0]
    exp = tf.reverse(tf.exp(prediction), axis=[0])
    partial_sum_a = cumsum(exp, n_observations)
    partial_sum = tf.reverse(partial_sum_a, axis=[0]) + 1
    log_at_risk = tf.log(tf.gather(partial_sum, tf.reshape(at_risk_label, [-1])) + 1e-50)
    diff = prediction - log_at_risk
    times = tf.reshape(diff, [-1]) * observed
    cost = - (tf.reduce_sum(times))
    return cost


def cumsum(x, observations):
    """The function for calculating cumulative sumation vector.

    This function receives a vector input and calculates its cumulative function
    representation as another vector.

    Args
    ----
        x : numpy float32 representing the vector input
        observations : int  representing the length of x vector.

    Returns
    -------
        cumsum : numpy float32 vector representing the cumulated sum of the input.

    """

    x = tf.reshape(x, (1, observations))
    # values = tf.split(1, x.get_shape()[1], x)
    values = tf.split(x, x.get_shape()[1], 1)
    out = []
    prev = tf.zeros_like(values[0])
    for val in values:
        s = prev + val
        out.append(s)
        prev = s
    cumsum = tf.concat(out, 1)
    cumsum = tf.reshape(cumsum, (observations, 1))
    return cumsum


class NanLossHook(tf.train.SessionRunHook):
    r"""Automate the NaN tensor hook.

    Monitors the loss tensor and stops training if loss is NaN.
    Can either fail with exception or just stop training.

    Example
    -------
    ...
    hooks = [NanLossHook(loss, fail_on_nan_loss=True)]
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        ...
        pass
        ...

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


def main(args):
    """The function for TF-Slim DNN model training.
    This function receives user-given parameters as gflag arguments. Then it
    creates the tensorflow model-graph, defines loss and optimizer. Finally,
    creates a training loop and saves the results and logs in the sub-directory.
    Args:
        args: This brings all gflags given user inputs with default values.
    Returns:
        None
    """

    data_x, data_y, c = data_providers(FLAGS.x_data, FLAGS.y_data)

    X = data_x
    C = c
    T = data_y

    n = FLAGS.split_ratio
    fold = int(len(X) / 10)
    train_set = {}
    test_set = {}
    final_set = {}

    sa = SurvivalAnalysis()
    train_set['X'], train_set['T'], train_set['C'], train_set['A'] = sa.calc_at_risk(X[0:fold * 6, ], T[0:fold * 6], C[0:fold * 6]);

    n_obs = train_set['X'].shape[0]
    n_features = train_set['X'].shape[1]
    observed = 1 - train_set['C']

    # Start building the graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        logging.set_verbosity(tf.logging.INFO)

        if not tf.gfile.Exists(FLAGS.ckpt_dir):
            tf.gfile.MakeDirs(FLAGS.ckpt_dir)

        n_batches = n_obs / FLAGS.batch_size
        decay_steps = int(FLAGS.n_epochs * n_batches)

        # x_batch, at_risk_batch, observed_batch = tf.train.shuffle_batch(
        #     [train_set['X'],
        #      train_set['A'],
        #      observed],
        #     batch_size=FLAGS.batch_size,
        #     capacity=50000,
        #     min_after_dequeue=10000,
        #     num_threads=1,
        #     allow_smaller_final_batch=True)

        global_step = get_or_create_global_step()

        lr = tf.train.exponential_decay(learning_rate=FLAGS.init_lr,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=FLAGS.lr_decay_rate,
                                        staircase=True)

        # ======== I changed these parameters to make the model diverge ========
        FLAGS.beta = 1.0
        lr = 10.0
        FLAGS.n_neurons = 10
        FLAGS.n_layers = 4
        # ======================================================================

        # Create the model and pass the input values batch by batch
        hidden_layers = [FLAGS.n_neurons] * FLAGS.n_layers

        pred, end_points = multilayer_nn_model(train_set['X'],  # x_batch
                                               hidden_layers,
                                               FLAGS.n_classes,
                                               FLAGS.beta)

        # Define loss
        cost = estimate_cost(pred, train_set['A'], observed)
        tf.losses.add_loss(cost)
        total_loss = tf.losses.get_total_loss()

        tf.summary.scalar('loss', total_loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        # create the back-propagation object
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            # clip_gradient_norm=4,   # Gives quick convergence
            # Commented this clip_gradient_norm, because it supports model to converge.
            check_numerics=False,
            summarize_gradients=True)

        hooks = [NanLossHook(total_loss, fail_on_nan_loss=True)]
        with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:

            while not sess.should_stop():
                sess.run(train_op)

            # create the training loop
            final = slim.learning.train(
                train_op,
                FLAGS.ckpt_dir,
                init_op=tf.global_variables_initializer(),  # Or set to 'None'
                log_every_n_steps=1,
                graph=graph,
                global_step=global_step,
                number_of_steps=FLAGS.n_epochs,
                save_summaries_secs=20,
                startup_delay_steps=0,
                saver=None,
                save_interval_secs=10,
                trace_every_n_steps=1,
            )


if __name__ == '__main__':
    tf.app.run()
