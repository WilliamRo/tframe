from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import context
from tframe import hub as th
from tframe import pedia


class Optimizer(object):
  """Tframe optimizer class allowing more sophisticated operations on gradients.
  For example, gradient clipping, and learning rate schedule.
  """

  def __init__(self, tf_optimizer):
    assert isinstance(tf_optimizer, tf.train.Optimizer)
    self.tf_optimizer = tf_optimizer

    self.reset_tf_optimizer = None


  def minimize(self, loss, var_list=None):
    # Step 1: compute gradients
    grads_and_vars = self._compute_gradients(loss, var_list=var_list)
    # Step 2: apply gradients
    update = self.tf_optimizer.apply_gradients(grads_and_vars)
    # Set reset_tf_optimizer if necessary
    if th.reset_optimizer_after_resurrection and th.lives > 0:
      self.reset_tf_optimizer = tf.variables_initializer(
        self.tf_optimizer.variables())
    return update


  def _compute_gradients(self, loss, var_list=None):
    # Sanity check
    assert isinstance(loss, tf.Tensor)

    # Compute gradients using default method
    grads_and_vars = self.tf_optimizer.compute_gradients(
      loss, var_list=var_list)

    # Deal with Nan if required
    if th.opt_nan_protection: grads_and_vars = [
      (self.deal_with_nan(grad), var) for grad, var in grads_and_vars]

    # Modify learning rate after resurrection if necessary
    res_decay = th.opt_lr_multiplier
    if res_decay < 1.0:
      assert res_decay > 0
      grads_and_vars = [(grad * res_decay, var) for grad, var in grads_and_vars]

    # Clip gradients if necessary
    if th.clip_threshold > 0:
      grads_and_vars = self.clip_gradients(grads_and_vars)

    return grads_and_vars


  # region: Static and class methods

  @staticmethod
  def deal_with_nan(grad):
    # Reference: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
    assert isinstance(grad, tf.Tensor)
    return tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad)

  @staticmethod
  def clip_gradients(grads_and_vars):
    """This method was migrated from GradientClipOptimizer which has been
       deprecated"""
    clip_method = th.clip_method
    bound = th.clip_threshold
    assert clip_method in ('norm', 'value', 'global_norm', 'avg_norm')

    if clip_method in ('norm', 'value', 'avg_norm'):
      if clip_method == 'norm':
        method = lambda g: tf.clip_by_norm(g, bound)
      elif clip_method == 'value':
        method = lambda g: tf.clip_by_value(g, -bound, bound)
      else:
        method = lambda g: tf.clip_by_average_norm(g, bound)
      grads_and_vars = [(method(grad), var) for grad, var in grads_and_vars]
    else:
      assert clip_method == 'global_norm'
      grads = [g for g, _ in grads_and_vars]
      clipped_grads, _ = tf.clip_by_global_norm(grads, bound)
      vars_ = [v for _, v in grads_and_vars]
      grads_and_vars = list(zip(clipped_grads, vars_))

    return grads_and_vars

  @classmethod
  def get_optimizer(cls, identifier):
    """Get tframe optimizer from an identifier, which can be
       (1) a string or a class, e.g., `adam` or tf.train.Adam
       (2) an instance of tensorflow optimizer, e.g., tf.train.Adam()
       (3) a tframe optimizer
    """
    # Case (3)
    if isinstance(identifier, Optimizer): return identifier
    # Case (1)
    optimizer = cls.get_tf_optimizer(identifier)
    # Case (2)
    if not isinstance(optimizer, tf.train.Optimizer):
      raise TypeError('!! Failed to get optimizer')

    # Set tensorflow optimizer to context for future use
    context.tf_optimizer = optimizer
    # The code below is migrated from previous version
    if not th.save_train_opt_vars: optimizer._name = pedia.train_opt

    # Wrap and return
    return Optimizer(optimizer)

  @classmethod
  def get_tf_optimizer(cls, optimizer):
    if isinstance(optimizer, tf.train.Optimizer): return optimizer

    if optimizer in ['adam', tf.train.AdamOptimizer]:
      return tf.train.AdamOptimizer(
        learning_rate=th.learning_rate,
        beta1=th.beta1, beta2=th.beta2, epsilon=th.adam_epsilon)
    elif optimizer in ['rmsprop', tf.train.RMSPropOptimizer]:
      return tf.train.RMSPropOptimizer(
        learning_rate=th.learning_rate)
    return optimizer(th.learning_rate)

  # endregion: Static and class methods

