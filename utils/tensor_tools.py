from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf


def extract_last_wrapper(op):
  """This method should only be used for training RNN models. Input tensors
     should have the shape (batch, step, *dim)"""
  def _op(tensor1, tensor2, *args, **kwargs):
    assert isinstance(tensor1, tf.Tensor) and isinstance(tensor2, tf.Tensor)
    return op(tensor1[:, -1], tensor2[:, -1], *args, **kwargs)
  return _op

