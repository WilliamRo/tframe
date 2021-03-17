from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tframe import checker
from tframe import hub


class GradientClipOptimizer(object):
  def __init__(self, tf_optimizer, threshold, method='norm'):
    assert isinstance(tf_optimizer, tf.train.Optimizer)
    self._tf_optimizer = tf_optimizer
    self._threshold = checker.check_type(threshold, float)
    assert threshold >= 0
    self._method = method
    assert method in ('norm', 'global_norm', 'value', 'avg_norm')

    self.reset_tf_optimizer = None

  # region : Public Methods

  def minimize(self, loss, var_list=None):
    # Step 1: compute gradients
    grads_and_vars = self._compute_gradients(loss, var_list=var_list)
    # Step 2: apply gradients
    update = self._tf_optimizer.apply_gradients(grads_and_vars)
    # Set reset_tf_optimizer if necessary
    if hub.reset_optimizer_after_resurrection and hub.lives > 0:
      self.reset_tf_optimizer = tf.variables_initializer(
        self._tf_optimizer.variables())
    return update

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _deal_with_nan(grad):
    # Reference: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
    assert isinstance(grad, tf.Tensor)
    return tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad)

  def _compute_gradients(self, loss, var_list=None):
    # Sanity check
    assert isinstance(loss, tf.Tensor)

    # Compute gradients using default method
    grads_and_vars = self._tf_optimizer.compute_gradients(
      loss, var_list=var_list)

    # Deal with NaN if necessary
    if hub.clip_nan_protection: grads_and_vars = [
      (self._deal_with_nan(grad), var) for grad, var in grads_and_vars]

    # Apply lr decay if necessary
    lr_decay = hub.clip_lr_multiplier
    if lr_decay < 1.0:
      assert lr_decay > 0
      grads_and_vars = [(grad * lr_decay, var) for grad, var in grads_and_vars]

    # Clip gradient if necessary
    if self._threshold > 0:
      bound = self._threshold
      if self._method in ('norm', 'value', 'avg_norm'):
        if self._method == 'norm':
          method = lambda g: tf.clip_by_norm(g, bound)
        elif self._method == 'value':
          method = lambda g: tf.clip_by_value(g, -bound, bound)
        else: method = lambda g: tf.clip_by_average_norm(g, bound)
        grads_and_vars = [(method(grad), var) for grad, var in grads_and_vars]
      else:
        assert self._method == 'global_norm'
        grads = [g for g, _ in grads_and_vars]
        clipped_grads, _ = tf.clip_by_global_norm(grads, self._threshold)
        vars_ = [v for _, v in grads_and_vars]
        grads_and_vars = list(zip(clipped_grads, vars_))

    return grads_and_vars

  # endregion : Private Methods


