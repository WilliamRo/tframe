from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import console
from tframe import context
from tframe import hub
from tframe import monitor

from tframe.utils.maths.stat_tools import Statistic

from .etch_kernel import EtchKernel


class Cola(EtchKernel):

  def __init__(self, weights, ratio):
    # Call parent's constructor
    super().__init__(weights)

    # assert isinstance(ratio, float) and 0 < ratio < 1
    self.ratio = ratio
    self.max_flip = hub.max_flip


  def _get_new_mask(self):

    assert hub.monitor_weight_flips
    flips = monitor.get_weight_flip_count(self.weights)
    mask = (flips < self.max_flip).astype(float)
    if hub.flip_irreversible: mask *= self.mask_buffer

    return mask


  # # Back up
  # def _get_new_mask(self):
  #   # Get corresponding grads stats
  #   assert hub.monitor_weight_grads
  #   # When monitor_weight_grads is True, grads stats of self.weights will be
  #   # .. monitored and can be accessed
  #   grad_stats = monitor.get_weight_stats(self.weights)
  #   assert isinstance(grad_stats, Statistic)
  #
  #   # Get absolute running average of gradient and current weights
  #   graa = grad_stats.running_abs_average
  #   # graa = np.abs(grad_stats.running_average)
  #
  #   wa = np.abs(self.weights_buffer)
  #   assert isinstance(graa, np.ndarray) and isinstance(wa, np.ndarray)
  #   assert graa.shape == wa.shape
  #
  #   # Get percentile
  #   pctile = 100 - self.weights_fraction * (1 - self.ratio)
  #
  #   g_mask = graa < np.percentile(graa, pctile)
  #   # g_mask = graa > np.percentile(graa, 100 - pctile)
  #
  #   # w_mask = wa < np.percentile(wa, pctile)
  #   w_mask = wa > np.percentile(wa, 100 - pctile)
  #
  #   mask = self.mask_buffer
  #   mask[g_mask * w_mask] = 0.
  #
  #   return mask
  #
