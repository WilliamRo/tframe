from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tframe import checker, context
from tframe.core.quantity import Quantity


def _flatten(tensor):
  assert isinstance(tensor, tf.Tensor)
  shape = tensor.shape.as_list()
  if len(shape) > 2:
    tensor = tf.reshape(tensor, (-1, shape[-1]))
  else: assert len(shape) == 2
  return tensor


def sigmoid_cross_entropy(labels, outputs):
  # Calculate average cross-entropy
  with tf.name_scope('binary_cross_entropy'):
    if context.logits_tensor is not None:
      # assert outputs is context.logits_tensor
      return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=outputs)
    else:
      xent = -tf.reduce_sum(labels * tf.log(outputs + 1e-6), 1)
      return xent


def cross_entropy(labels, outputs):
  # Calculate average cross-entropy
  with tf.name_scope('cross_entropy'):
    # TODO: to be refactored
    if context.logits_tensor is not None:
      # assert outputs is context.logits_tensor
      return tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=outputs)
    else:
      xent = -tf.reduce_sum(labels * tf.log(outputs + 1e-6), 1)
      return xent


def mean_squared_error(y_true, y_predict):
  return tf.square(y_true - y_predict)


def euclidean(y_true, y_predict):
  distances = tf.norm(y_true - y_predict, axis=-1)
  return distances


def get(identifier, last_only=False, use_logits=None, **kwargs):
  if isinstance(identifier, Quantity): return identifier
  elif callable(identifier):
    # Metrics got in this way do not support batch validation
    return Quantity(identifier)

  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    # tr_summ_method is set to tf.reduce_mean by default
    kernel, tf_summ_method, np_summ_method = None, tf.reduce_mean, None

    if identifier in ['mean_squared', 'mean_squared_error', 'mse']:
      kernel = mean_squared_error
    elif identifier in ['cross_entropy', 'softmax_cross_entropy']:
      if use_logits is None: use_logits = True
      kernel = cross_entropy
    elif identifier in ['sigmoid_cross_entropy', 'binary_cross_entropy']:
      if use_logits is None: use_logits = True
      kernel = sigmoid_cross_entropy
    elif identifier in ['euclid', 'euclidean']: kernel = euclidean
    else: raise ValueError('Can not resolve `{}`'.format(identifier))

    # Set use_logits to False by default
    if use_logits is None: use_logits = False
    return Quantity(kernel, tf_summ_method, np_summ_method, last_only,
                    name='Loss', use_logits=use_logits, **kwargs)
  else: raise TypeError('identifier must be a Quantity, function or a string')

