from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .etch_kernel import EtchKernel


class Lottery(EtchKernel):

  def __init__(self, weights, prune_frac):
    # Call parent's constructor
    super().__init__(weights)

    assert isinstance(prune_frac, float) and 0 < prune_frac < 1
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

