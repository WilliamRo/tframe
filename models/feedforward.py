from __future__ import absolute_import

import tensorflow as tf

from .model import Model
from .model import Predictor

from ..nets import Net

from .. import losses
from ..utils.local import clear_paths


class Feedforward(Model, Net, Predictor):
  """Feedforward network, also known as multilayer perceptron"""
  def __init__(self, mark=None):
    Model.__init__(self, mark)
    Net.__init__(self, 'FeedforwardNet')
    Predictor.__init__(self)

  def build(self, loss='cross_entropy', optimizer=None):
    # Feed forward to get outputs and set targets placeholder
    self.outputs = self()
    self.targets = tf.placeholder(tf.float32, self.outputs.get_shape(),
                                  name='targets')
    # Get loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      self._loss = loss_function(self.targets, self.outputs)
    # Get train step
    self._optimizer = (tf.train.AdamOptimizer(0.01) if optimizer is None
                       else optimizer)
    self._train_step = self._optimizer.minimize(self._loss)


    self._session = tf.Session()
    log_dir = self.log_dir
    clear_paths(log_dir)
    self._summary_writer = tf.summary.FileWriter(log_dir, self._session.graph)
