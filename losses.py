from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
from tframe import tf

from tframe import checker, context, linker
from tframe.core.quantity import Quantity
from tframe.utils.arg_parser import Parser


# region : Private

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

# endregion : Private

# region : Major Losses

def sigmoid_cross_entropy(labels, outputs):
  # Calculate average cross-entropy
  with tf.name_scope('binary_cross_entropy'):
    use_logits = outputs in context.logits_tensor_dict.values()
    if use_logits:
      # assert outputs is context.logits_tensor
      return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=outputs + _epsilon)
    else:
      xent = -tf.reduce_sum(labels * tf.log(outputs + _epsilon), axis=-1)
      return xent


def cross_entropy(labels, outputs):
  use_logits = outputs in context.logits_tensor_dict.values()
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
  use_logits = logits in context.logits_tensor_dict.values()
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


def _weighted_mxe(y_true: tf.Tensor, y_predict: tf.Tensor, tf_func, min_w):
  """Calculate MSE or MAE using y_true as weights. assert y_true \in [0, 1]"""
  from tframe import hub as th

  assert tf_func in (tf.square, tf.abs)
  assert 0 <= min_w <= 1

  reduce_axis = list(range(1, len(y_true.shape)))
  weights = tf.maximum(y_true, min_w)

  nume = tf.reduce_sum(tf_func(y_true - y_predict) * weights, axis=reduce_axis)
  deno = tf.reduce_sum(weights, axis=reduce_axis)
  return tf.divide(nume, deno)


def weighted_mse(y_true: tf.Tensor, y_predict: tf.Tensor, min_w=0):
  return _weighted_mxe(y_true, y_predict, tf.square, min_w)


def weighted_mae(y_true: tf.Tensor, y_predict: tf.Tensor, min_w=0):
  return _weighted_mxe(y_true, y_predict, tf.abs, min_w)


def mean_balanced_error(y_true: tf.Tensor, y_predict: tf.Tensor):
  """"""
  reduce_axis = list(range(1, len(y_true.shape)))

  # Define similarity ratio
  magic_number = 1e-6
  _min = tf.minimum(y_true, y_predict)
  _max = tf.maximum(y_true, y_predict)
  sr = _min / tf.maximum(_max, magic_number)
  tp = sr * y_true

  # Calculate F1 score
  be = 1 - 2 * tp / (y_true + y_predict + _epsilon)

  return tf.reduce_mean(be, reduce_axis)


def global_balanced_error(y_true: tf.Tensor, y_predict: tf.Tensor):
  reduce_axis = list(range(1, len(y_true.shape)))

  _epsilon = 1e-6

  # Define similarity ratio
  # _min = tf.minimum(y_true, y_predict) + _epsilon
  # _max = tf.maximum(y_true, y_predict) + _epsilon
  # sr = _min / _max
  _min = tf.minimum(y_true, y_predict)
  _max = tf.maximum(y_true, y_predict)
  sr = _min / tf.maximum(_max, 1e-8)
  tp = tf.reduce_sum(sr * y_true, axis=reduce_axis) + _epsilon

  # Calculate F1 score
  return 1 - 2 * tp / tf.reduce_sum(
    y_true + y_predict + 2 * _epsilon, axis=reduce_axis)

  # # Define similarity ratio
  # _min = tf.minimum(y_true, y_predict)
  # _max = tf.maximum(y_true, y_predict)
  # sr = _min / tf.maximum(_max, 1e-6)
  # tp = tf.reduce_sum(sr * y_true, axis=reduce_axis)
  #
  # # Calculate F1 score
  # return 1 - 2 * tp / tf.reduce_sum(y_true + y_predict, axis=reduce_axis)


def mean_absolute_error(y_true, y_predict):
  return tf.abs(y_true - y_predict)


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

# endregion : Major Losses

# region : Auxiliary Losses

def saturate_loss(tensor, mu=0.5, encourage_saturation=True):
  # TODO: this method is still being developed. DO NOT USE WITHOUT GUIDE
  # Each entry in tensor should be in range [0, 1], which should be guaranteed
  # ... by users
  checker.check_type(tensor, tf.Tensor)
  assert 0 < mu < 1
  # Encourage saturation
  if encourage_saturation:
    # Calculate distance to saturation
    left = tensor[tf.less(tensor, mu)]
    right = tensor[tf.greater(tensor, 0.5)]
    return tf.norm(left) - tf.norm(right)
  else:
    # Calculate distance to unsaturation
    degree = tf.abs(tensor - mu)
  # Calculate loss using reduce mean
  return tf.reduce_mean(degree)

# endregion : Auxiliary Losses


def get(identifier, last_only=False, **kwargs):
  # TODO: use_logits parameter has not been used yet

  if isinstance(identifier, Quantity): return identifier
  elif callable(identifier):
    # Metrics got in this way do not support batch validation
    return Quantity(identifier)

  elif isinstance(identifier, six.string_types):
    # Parse identifier
    p = Parser.parse(identifier)
    identifier = p.name.lower()

    # tr_summ_method is set to tf.reduce_mean by default
    kernel, tf_summ_method, np_summ_method = None, tf.reduce_mean, None
    use_logits = False

    if identifier in ['mean_squared', 'mean_squared_error', 'mse']:
      kernel = mean_squared_error
    elif identifier in ['mean_absolute', 'mean_absolute_error', 'mae']:
      kernel = mean_absolute_error
    elif identifier in ['mean_balanced_error', 'mbe']:
      kernel = mean_balanced_error
    elif identifier in ['global_balanced_error', 'gbe', 'ber']:
      kernel = global_balanced_error
    elif identifier in ['weighted_mean_squared_error', 'wmse']:
      min_w = p.get_arg(float)
      kernel = lambda *args: weighted_mse(*args, min_w=min_w)
    elif identifier in ['rmse', 'root_mean_squared_error']:
      kernel = mean_squared_error
      def tf_rmse_summ(x): return tf.sqrt(tf.reduce_mean(x))
      def np_rmse_summ(x): return np.sqrt(np.mean(x))
      tf_summ_method, np_summ_method = tf_rmse_summ, np_rmse_summ
    elif identifier in ['weighted_mean_absolute_error', 'wmae']:
      min_w = p.get_arg(float)
      kernel = lambda *args: weighted_mae(*args, min_w=min_w)
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
    if 'name' not in kwargs: kwargs['name'] = 'Loss'
    return Quantity(kernel, tf_summ_method, np_summ_method, last_only,
                    use_logits=use_logits, **kwargs)
  else:
    raise TypeError('identifier must be a Quantity, function or a string.'
                    f' `{identifier}` is illegal.')

