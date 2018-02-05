from __future__ import absolute_import

import six

import tensorflow as tf


def accuracy(labels, outputs):
  assert isinstance(labels, tf.Tensor) and isinstance(outputs, tf.Tensor)
  label_shape = labels.get_shape().as_list()
  if len(label_shape) > 1 or label_shape[1] > 1:
    labels = tf.argmax(labels, 1, name='labels')
    outputs = tf.argmax(outputs, 1, name='prediction')

  correct_prediction = tf.equal(labels, outputs, 'correct_prediction')

  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                         name='accuracy')


def delta(truth, output):
  assert isinstance(truth, tf.Tensor) and isinstance(output, tf.Tensor)
  return tf.norm(truth - output)


def norm_error_ratio(truth, output):
  assert isinstance(truth, tf.Tensor) and isinstance(output, tf.Tensor)
  return tf.norm(truth - output) / tf.norm(truth) * 100


def rms_error_ratio(truth, output):
  assert isinstance(truth, tf.Tensor) and isinstance(output, tf.Tensor)
  rms = lambda x: tf.sqrt(tf.reduce_mean(tf.square(x)))
  return rms(truth - output) / rms(truth) * 100


def get(identifier):
  if identifier is None or callable(identifier):
    return identifier
  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    if identifier in ['accuracy', 'acc']:
      return accuracy
    elif identifier in ['delta', 'distance']:
      return delta
    elif identifier in ['ratio', 'norm_ratio']:
      return norm_error_ratio
    elif identifier in ['rms_ratio']:
      return rms_error_ratio
    else:
      raise ValueError('Can not resolve "{}"'.format(identifier))
  else:
    raise TypeError('identifier must be a function or a string')
