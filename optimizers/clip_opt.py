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

  # region : Public Methods

  def minimize(self, loss, var_list=None):
    # Step 1: compute gradients
    grads_and_vars = self._compute_gradients(loss, var_list=var_list)
    # Step 2: apply gradients
    return self._tf_optimizer.apply_gradients(grads_and_vars)

  # endregion : Public Methods

  # region : Private Methods

  def _compute_gradients(self, loss, var_list=None):
    # Sanity check
    assert isinstance(loss, tf.Tensor)

    # Compute gradients using default method
    grads_and_vars = self._tf_optimizer.compute_gradients(
      loss, var_list=var_list)

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


