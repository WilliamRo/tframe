from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tframe import checker


class GradientClipOptimizer(object):
  def __init__(self, tf_optimizer, threshold):
    assert isinstance(tf_optimizer, tf.train.Optimizer)
    self._tf_optimizer = tf_optimizer
    self._threshold = checker.check_type(threshold, float)
    assert threshold >= 0

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
      grads_and_vars = [
        (tf.clip_by_value(grad, -self._threshold, self._threshold), var)
        for grad, var in grads_and_vars]

    return grads_and_vars

  # endregion : Private Methods


