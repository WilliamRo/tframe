from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tframe import checker


def _flatten(tensor):
  assert isinstance(tensor, tf.Tensor)
  shape = tensor.shape.as_list()
  if len(shape) > 2:
    tensor = tf.reshape(tensor, (-1, shape[-1]))
  else: assert len(shape) == 2
  return tensor


def _extract_last(labels, logits):
  """ Extract last output for sequence classification task.
      Currently can not be used with parallel engine.
      :param labels & logits: tensors of with shape
                              [batch_size(=1), num_steps, *shape]
  """
  assert isinstance(labels, tf.Tensor) and isinstance(logits, tf.Tensor)
  # tf.assert_equal(labels.shape[0], 1)
  # tf.assert_equal(logits.shape[0], 1)
  return labels[0, -1], logits[0, -1]


def sigmoid_cross_entropy(labels, logits, last_only=False):
  checker.check_tensor_shape(labels, logits, 'labels', 'logits')
  if last_only: labels, logits = _extract_last(labels, logits)
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


def cross_entropy(labels, logits, last_only=False):
  # Make sure labels and logits has a same shape
  checker.check_tensor_shape(labels, logits, 'labels', 'logits')
  if last_only: labels, logits = _extract_last(labels, logits)
  # Convert labels and logits to 2-D tensors
  # tensors = [labels, logits]
  # TODO: no need to flatten
  # for i, tensor in enumerate(tensors):
  #   tensors[i] = _flatten(tensor)
  # Calculate average cross-entropy
  with tf.name_scope('cross_entropy'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits))


def mean_squared_error(y_true, y_predict, last_only=False):
  if last_only: y_true, y_predict = _extract_last(y_true, y_predict)
  return tf.reduce_mean(tf.square(y_true - y_predict))
  # return tf.reduce_mean(tf.square(tf.abs(y_true - y_predict)))


def euclidean(y_true, y_predict):
  distances = tf.norm(y_true - y_predict)
  return tf.reduce_mean(distances)


def get(identifier):
  if callable(identifier):
    return identifier
  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    if identifier in ['mean_squared', 'mean_squared_error', 'mse']:
      return mean_squared_error
    elif identifier in ['cross_entropy', 'softmax_cross_entropy']:
      return cross_entropy
    elif identifier in ['sigmoid_cross_entropy', 'binary_cross_entropy']:
      return sigmoid_cross_entropy
    elif identifier in ['euclid', 'euclidean']:
      return euclidean
    else:
      raise ValueError('Can not resolve "{}"'.format(identifier))
  else:
    raise TypeError('identifier must be a function or a string')

