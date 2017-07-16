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


def get(identifier):
  if identifier is None or callable(identifier):
    return identifier
  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    if identifier in ['accuracy', 'acc']:
      return accuracy
    else:
      raise ValueError('Can not resolve "{}"'.format(identifier))
  else:
    raise TypeError('identifier must be a function or a string')
