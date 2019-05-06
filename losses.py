from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tframe import checker, context
from tframe.utils.tensor_tools import extract_last_wrapper


def _flatten(tensor):
  assert isinstance(tensor, tf.Tensor)
  shape = tensor.shape.as_list()
  if len(shape) > 2:
    tensor = tf.reshape(tensor, (-1, shape[-1]))
  else: assert len(shape) == 2
  return tensor


def sigmoid_cross_entropy(labels, logits):
  checker.check_tensor_shape(labels, logits, 'labels', 'logits')
  # Convert labels and logits to 2-D tensors
  # tensors = [labels, logits]
  # for i, tensor in enumerate(tensors):
  #   tensors[i] = _flatten(tensor)
  # Calculate average cross-entropy
  with tf.name_scope('binary_cross_entropy'):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits))
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #   labels=tensors[0], logits=tensors[1]))


def cross_entropy(labels, outputs):
  # Make sure labels and logits has a same shape
  checker.check_tensor_shape(labels, outputs, 'labels', 'logits')
  # Convert labels and logits to 2-D tensors
  # tensors = [labels, logits]
  # TODO: no need to flatten
  # for i, tensor in enumerate(tensors):
  #   tensors[i] = _flatten(tensor)
  # Calculate average cross-entropy
  with tf.name_scope('cross_entropy'):
    # TODO: to be refactored
    if context.logits_tensor is not None:
      # assert outputs is context.logits_tensor
      return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=outputs))
    else:
      xent = -tf.reduce_sum(labels * tf.log(outputs + 1e-6), 1)
      return tf.reduce_mean(xent)


def mean_squared_error(y_true, y_predict):
  return tf.reduce_mean(tf.square(y_true - y_predict))


def euclidean(y_true, y_predict):
  distances = tf.norm(y_true - y_predict)
  return tf.reduce_mean(distances)


def get(identifier, last_only=False):
  if callable(identifier):
    return identifier
  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    if identifier in ['mean_squared', 'mean_squared_error', 'mse']:
      f =  mean_squared_error
    elif identifier in ['cross_entropy', 'softmax_cross_entropy']:
      f =  cross_entropy
    elif identifier in ['sigmoid_cross_entropy', 'binary_cross_entropy']:
      f =  sigmoid_cross_entropy
    elif identifier in ['euclid', 'euclidean']:
      f =  euclidean
    else:
      raise ValueError('Can not resolve "{}"'.format(identifier))

    if last_only: return extract_last_wrapper(f)
    else: return f
  else:
    raise TypeError('identifier must be a function or a string')

