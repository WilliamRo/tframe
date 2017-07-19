from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_scale(tensor):
  assert isinstance(tensor, tf.Tensor)
  return tensor.get_shape().as_list()[1:]


def shape_string(shape):
  assert isinstance(shape, list)
  strs = [str(x) if x is not None else "?" for x in shape]
  return "x".join(strs)
