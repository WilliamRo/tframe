from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import tframe as tfr


def _truncate(truth, output):
  # TODO: only supported for some metrics
  assert len(truth.shape.as_list()) > 2
  i = tf.cond(tf.get_collection(tfr.pedia.is_training)[0],
              lambda: 0, lambda: tfr.hub.val_preheat)
  return truth[:, i:], output[:, i:]


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
  if tfr.hub.val_preheat > 0:
    truth, output = _truncate(truth, output)
  return tf.norm(truth - output)


def norm_error_ratio(truth, output):
  assert isinstance(truth, tf.Tensor) and isinstance(output, tf.Tensor)
  if tfr.hub.val_preheat > 0:
    truth, output = _truncate(truth, output)
  return tf.norm(truth - output) / tf.norm(truth) * 100


def rms_error_ratio(truth, output):
  assert isinstance(truth, tf.Tensor) and isinstance(output, tf.Tensor)
  rms = lambda x: tf.sqrt(tf.reduce_mean(tf.square(x)))
  # TODO: pilot, tfr.hub.val_preheat > 0 only happens in RNN model
  #       thus output.shape is [batch_size, step_num, *target_shape]
  if tfr.hub.val_preheat > 0:
    truth, output = _truncate(truth, output)
  return rms(truth - output) / rms(truth) * 100


def get(identifier):
  if callable(identifier):
    return identifier
  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    if identifier in ['accuracy', 'acc']:
      f = accuracy
    elif identifier in ['delta', 'distance']:
      f = delta
    elif identifier in ['ratio', 'norm_ratio']:
      f = norm_error_ratio
    elif identifier in ['rms_ratio']:
      f = rms_error_ratio
    else:
      raise ValueError('Can not resolve "{}"'.format(identifier))
    return f
  else:
    raise TypeError('identifier must be a function or a string')



