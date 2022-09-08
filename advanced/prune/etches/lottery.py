from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import hub
from tframe import monitor

from tframe.utils.maths.stat_tools import Statistic

from .etch_kernel import EtchKernel


class Lottery(EtchKernel):

  def __init__(self, weights, prune_frac):
    # Call parent's constructor
    super().__init__(weights)

    assert isinstance(prune_frac, float) and 0 <= prune_frac < 1
    self.prune_frac = prune_frac

    # Variable and ops used for resetting
    self.init_val = tf.Variable(
      tf.zeros_like(weights), trainable=False, name='init_val')
    # Define assign ops
    self.assign_init = tf.assign(self.init_val, self.weights)
    self.reset_weights = tf.assign(self.weights, self.init_val)


  def _get_new_mask(self):
    """self.prune_frac should be global prune_rate_fc * kernel_prune_frac
       This value should be correctly set during KernelBase initialization
    """
    if hub.lottery_kernel == 'lottery18': return self._lottery18()
    elif hub.lottery_kernel == 'g_constraint': return self._g_constraint()
    else: raise KeyError(
      '!! Unknown lottery kernel `{}`'.format(hub.lottery_kernel))


  def _g_constraint(self):
    # Get corresponding grads stats
    assert hub.monitor_weight_grads
    # When monitor_weight_grads is True, grads stats of self.weights will be
    # .. monitored and can be accessed
    grad_stats = monitor.get_weight_stats(self.weights)
    assert isinstance(grad_stats, Statistic)
    graa = grad_stats.running_abs_average

    # p is the fraction to prune
    p = self.prune_frac
    # Get mask and weights
    m_float = self.mask_buffer
    m_bool = np.asarray(m_float, np.bool)
    abs_w = np.abs(self.weights_buffer * self.mask_buffer)

    # Get mask according to |w| (remaining p smallest weights)
    w_bound = np.percentile(abs_w[m_bool], 100 * p)
    w_mask = abs_w < w_bound

    # Get gradient mask
    g_bound = np.percentile(graa[m_bool], 90)
    g_mask = graa < g_bound

    mask = self.mask_buffer
    mask[w_mask * g_mask * m_bool] = 0.0
    return mask


  def _lottery18(self):
    p = self.prune_frac
    w = self.weights_buffer * self.mask_buffer
    assert isinstance(w, np.ndarray) and np.isreal(self.weights_fraction)
    assert 0 < self.weights_fraction <= 100
    weight_fraction = self.weights_fraction * (1 - p)
    # Get weights magnitude
    w = np.abs(w)
    # Create mask
    mask = np.zeros_like(w, dtype=np.float32)
    mask[w > np.percentile(w, 100 - weight_fraction)] = 1.0
    # Return mask
    return mask


