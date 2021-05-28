from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf


class MaskedWeights(object):

  def __init__(self, weights, mask):
    assert isinstance(weights, tf.Variable)
    self.weights = weights
    assert isinstance(mask, (tf.Variable, tf.Tensor))
    self.mask = mask

    # Once weights and mask have been set, calculate masked weights
    self.masked_weights = tf.multiply(self.weights, self.mask, 'masked_weights')

    # Buffer
    self.weights_buffer = None
    self.mask_buffer = None
    # self.masked_weights_buffer = None


  @property
  def scope_abbr(self):
    """In tframe, RNN model, weight's name may be ../gdu/net_u/W"""
    scopes = self.weights.name.split('/')
    return '/'.join(scopes[1:-1])

  @property
  def weight_key(self):
    """Used in Pruner.extractor"""
    scopes = self.weights.name.split('/')
    key = '/'.join(scopes[1:])
    # key = key.split(':')[0]
    return key

  @property
  def mask_key(self):
    return self.weight_key + '_mask'


  def clear_buffers(self):
    self.weights_buffer = None
    self.mask_buffer = None
    # self.masked_weights_buffer = None




