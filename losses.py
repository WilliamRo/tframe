from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
import tensorflow as tf

from tframe import checker, context, linker
from tframe.core.quantity import Quantity


_epsilon = 1e-10

def _flatten(tensor):
  assert isinstance(tensor, tf.Tensor)
  shape = tensor.shape.as_list()
  if len(shape) > 2:
    tensor = tf.reshape(tensor, (-1, shape[-1]))
  else: assert len(shape) == 2
  return tensor


def _aligned(labels, outputs):
  assert isinstance(labels, tf.Tensor) and isinstance(outputs, tf.Tensor)
  return labels.shape.as_list()[-1] == outputs.shape.as_list()[-1]


def _reshape_labels(labels, num_classes=None):
  """Reshape labels for classification
  :param labels: a tensor of shape (d1, d2, ..., dn, 1)
  :param num_classes: if is None, the last dimension of labels will be removed
                      other wise labels will be reshaped to
                      (d1, ..., dn, num_classes)
  :return: the reshaped label
  """
  # currently tframe requires data points in data_dict of a DataSet keeps their
  #  dimension even if it is 1
  assert isinstance(labels, tf.Tensor) and labels.shape.as_list()[-1] == 1
  labels = tf.squeeze(labels, axis=-1)
  if num_classes is not None: labels = tf.one_hot(labels, num_classes)
  return labels


def sigmoid_cross_entropy(labels, outputs):
  # Calculate average cross-entropy
  with tf.name_scope('binary_cross_entropy'):
    if context.logits_tensor is not None:
      # assert outputs is context.logits_tensor
      return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=outputs + _epsilon)
    else:
      xent = -tf.reduce_sum(labels * tf.log(outputs + _epsilon), axis=-1)
      return xent


def cross_entropy(labels, outputs):
  use_logits = context.logits_tensor is not None
  # Calculate average cross-entropy
  with tf.name_scope('cross_entropy'):
    # TODO: to be refactored
    if use_logits:
      # TODO: not apply for RNN (due to while_loop)
      # assert outputs is context.logits_tensor
      if _aligned(labels, outputs):
        return tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=outputs + _epsilon)
      else:
        labels = _reshape_labels(labels, None)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=outputs + _epsilon)
    else:
      if not _aligned(labels, outputs):
        labels = _reshape_labels(labels, linker.get_dimension(outputs))
      xent = -tf.reduce_sum(labels * tf.log(outputs + _epsilon), axis=-1)
      return xent


def weighted_cross_entropy(labels, logits):
  """It is recommended that
     sum(class_weights) == len(logits.reshape(-1, num_classes))
  """
  from tframe import hub as th
  assert len(th.class_weights) == th.num_classes
  use_logits = context.logits_tensor is not None
  # Calculate weighted cross-entropy
  with tf.name_scope('weighted_cross_entropy'):
    if use_logits:
      if _aligned(labels, logits): raise NotImplementedError
      else:
        # Squeeze last dimension of sparse labels cuz tf only accept this shape
        labels = _reshape_labels(labels, None)
        weights = tf.gather(th.class_weights, labels)
        return tf.losses.sparse_softmax_cross_entropy(
          labels=labels,
          logits=logits + _epsilon,
          weights=weights,
        )
    else: raise NotImplementedError


def cross_entropy_base2(labels, outputs):
  xent = cross_entropy(labels, outputs)
  return tf.divide(xent, tf.log(2.))


def mean_squared_error(y_true, y_predict):
  return tf.square(y_true - y_predict)


def euclidean(y_true, y_predict):
  distances = tf.norm(y_true - y_predict, axis=-1)
  return distances


def tf_seq_loss_summ(x):
  # x.shape = [batch_size, num_steps]
  assert isinstance(x, tf.Tensor)
  shape = x.shape.as_list()
  assert len(shape) == 2
  return tf.reduce_mean(tf.reduce_sum(x, 1))


def np_seq_loss_summ(x):
  assert len(x.shape) == 2
  return np.mean(np.sum(x, axis=1))


def get(identifier, last_only=False, **kwargs):
  # TODO: use_logits parameter has not been used yet

  if isinstance(identifier, Quantity): return identifier
  elif callable(identifier):
    # Metrics got in this way do not support batch validation
    return Quantity(identifier)

  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    # tr_summ_method is set to tf.reduce_mean by default
    kernel, tf_summ_method, np_summ_method = None, tf.reduce_mean, None
    use_logits = False

    if identifier in ['mean_squared', 'mean_squared_error', 'mse']:
      kernel = mean_squared_error
    elif identifier in ['cross_entropy', 'softmax_cross_entropy']:
      use_logits = True
      kernel = cross_entropy
    elif identifier in ['wce', 'weighted_cross_entropy']:
      use_logits = True
      kernel = weighted_cross_entropy
      tf_summ_method, np_summ_method = None, None
    elif identifier in ['nlp_cross_entropy', 'nlp_softmax_cross_entropy']:
      use_logits = True
      kernel = cross_entropy
      tf_summ_method, np_summ_method = tf_seq_loss_summ, np_seq_loss_summ
    elif identifier in ['sigmoid_cross_entropy', 'binary_cross_entropy']:
      use_logits = True
      kernel = sigmoid_cross_entropy
    elif identifier in ['euclid', 'euclidean']: kernel = euclidean
    else: raise ValueError('Can not resolve `{}`'.format(identifier))

    # Set use_logits to False by default
    if use_logits is None: use_logits = False
    return Quantity(kernel, tf_summ_method, np_summ_method, last_only,
                    name='Loss', use_logits=use_logits, **kwargs)
  else: raise TypeError('identifier must be a Quantity, function or a string')

