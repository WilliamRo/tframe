from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf


def get_scale(tensor):
  assert isinstance(tensor, tf.Tensor)
  return tensor.get_shape().as_list()[1:]


def shape_string(input_):
  # If input is a single tensor
  if isinstance(input_, tf.Tensor):
    shapes = [get_scale(input_)]
  else:
    assert isinstance(input_, list) and len(input_) > 0
    if isinstance(input_[0], tf.Tensor):
      shapes = [get_scale(tensor) for tensor in input_]
    elif not isinstance(input_[0], list):
      shapes = [input_]

  result = ''
  for i, shape in zip(range(len(shapes)), shapes):
    result += ', ' if i > 0 else ''
    result += 'x'.join([str(x) if x is not None else "?" for x in shape])

  if len(shapes) > 1:
    result = '({})'.format(result)

  return result


def mark_str(value):
  if not isinstance(value, (tuple, list)): return ''
  result = ''
  for i, val in enumerate(value):
    if i > 0: result += '-'
    result += '{}'.format(val)
  return result


def get_name_by_levels(name, levels):
  assert isinstance(name, str)
  assert isinstance(levels, (list, tuple))
  scopes = name.split('/')
  scopes = [scope for i, scope in enumerate(scopes) if i in levels]
  return '/'.join(scopes)


def convert_to_one_hot(labels, num_classes):
  labels = np.array(labels)
  if len(labels.shape) < 2:
    sample_num = labels.shape[0]
    one_hot = np.zeros(shape=[sample_num, num_classes])
    one_hot[range(sample_num), labels] = 1
  else:
    one_hot = labels

  if len(one_hot.shape) != 2:
    raise ValueError('!! Input labels has an illegal dimension {}'.format(
      len(labels.shape)))

  return one_hot


def convert_to_dense_labels(one_hot):
  assert isinstance(one_hot, np.ndarray)
  if len(one_hot.shape) == 1: return one_hot
  assert len(one_hot.shape) == 2
  return np.argmax(one_hot, axis=1)


