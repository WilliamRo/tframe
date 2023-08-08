from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import inspect
import datetime
import math
from .np_tools import get_ravel_indices

from tframe import tf


def ordinal(n):
  return "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])


def get_scale(tensor):
  assert isinstance(tensor, tf.Tensor)
  return tensor.get_shape().as_list()[1:]


def shape_string(input_):
  # If input is a single tensor
  if isinstance(input_, tf.Tensor):
    shapes = [get_scale(input_)]
  else:
    assert isinstance(input_, (list, tuple)) and len(input_) > 0
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
  if isinstance(value, dict):
    result = ''
    for k, v in value.items():
      if result != '': result += '_'
      result += '{}{}'.format(k, v)
    return result

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
  assert num_classes > 1
  # Convert labels to numpy array with last dim larger than 1
  labels = np.array(labels)
  # if np.max(labels) == 1: return labels  # TODO: removed this line
  if labels.shape[-1] == 1: labels = labels.squeeze(axis=-1)
  # Prepare zeros
  label_shape = list(labels.shape)
  one_hot = np.zeros(shape=label_shape + [num_classes])
  indices = get_ravel_indices(labels) + (labels.ravel(),)
  one_hot[indices] = 1
  return one_hot

  # Remove the routines below when this method has been proved robust
  # labels = np.array(labels)
  # if len(labels.shape) < 2 or (len(labels.shape) == 2 and labels.shape[1] == 1):
  #   sample_num = labels.shape[0]
  #   one_hot = np.zeros(shape=[sample_num, num_classes])
  #   one_hot[range(sample_num), labels.flatten()] = 1
  # else:
  #   one_hot = labels
  #
  # if len(one_hot.shape) != 2:
  #   raise ValueError('!! Input labels has an illegal dimension {}'.format(
  #     len(labels.shape)))
  #
  # return one_hot


def convert_to_dense_labels(one_hot):
  """`one_hot` may be of shape [batch_size, 1, num_classes], which may appear
  in seq_set.summ_dict"""
  assert isinstance(one_hot, np.ndarray)
  # Convert targets in summ_dict if necessary
  if len(one_hot.shape) == 3 and one_hot.shape[1] == 1:
    one_hot = one_hot.reshape(one_hot.shape[0], -1)

  if len(one_hot.shape) == 1 or one_hot.shape[1] == 1: return one_hot

  # This occurs in situations that BatchReshape is used
  if len(one_hot.shape) != 2:
    one_hot = one_hot.reshape(-1, one_hot.shape[-1])

  assert len(one_hot.shape) == 2 and one_hot.shape[1] > 1
  return np.argmax(one_hot, axis=1)


def ravel_nested_stuff(nested_stuff, with_indices=False):
  # Sanity check
  assert isinstance(nested_stuff, (list, tuple))

  def ravel(stuff, base_indices=None):
    raveled_stuff = []
    indices_lists = []
    is_root = base_indices is None
    if is_root: base_indices = []
    for i, s in enumerate(stuff):
      leaf_index = base_indices + [i]
      if not isinstance(s, (tuple, list)):
        raveled_stuff.append(s)
        indices_lists.append(leaf_index)
      else:
        rs, il = ravel(s, leaf_index)
        raveled_stuff += rs
        indices_lists += il

    return raveled_stuff, indices_lists

  raveled_stuff, indices = ravel(nested_stuff)
  if with_indices: return raveled_stuff, indices
  else: return raveled_stuff


def transpose_tensor(tensor, perm):
  assert isinstance(tensor, tf.Tensor)
  perm += list(range(len(tensor.shape.as_list())))[len(perm):]
  return tf.transpose(tensor, perm)


def retrieve_name(var):
  """Gets the name of var. Does it from the out most frame inner-wards.
  Works only on Python 3.
  Reference: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string

  :param var: variable to get name from.
  :return: string
  """
  for fi in reversed(inspect.stack()):
    names = [var_name for var_name, var_val in fi.frame.f_locals.items()
             if var_val is var]
    if len(names) > 0: return names[0]
    else: return 'unknown_name'


def date_string(): return datetime.datetime.now().strftime('%m%d')

