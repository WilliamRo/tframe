from __future__ import absolute_import

import six

import tensorflow as tf


def cross_entropy(labels, logits):
  with tf.name_scope('cross_entropy'):
    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))


def mean_squared_error(y_true, y_predict):
  return tf.reduce_mean(tf.square(tf.abs(y_true - y_predict)))


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
    elif identifier in ['cross_entropy']:
      return cross_entropy
    elif identifier in ['euclid', 'euclidean']:
      return euclidean
    else:
      raise ValueError('Can not resolve "{}"'.format(identifier))
  else:
    raise TypeError('identifier must be a function or a string')

