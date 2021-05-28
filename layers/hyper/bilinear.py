from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub as th
from .hyper_base import HyperBase


class Bilinear(HyperBase):

  full_name = 'bilinear'
  abbreviation = 'bl'

  def __init__(
      self,
      dim1,
      dim2,
      activation=None,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      layer_normalization=False,
      max_norm=None,
      **kwargs):
    # Call parent's constructor
    super().__init__(activation, weight_initializer, use_bias,
                     bias_initializer, layer_normalization, **kwargs)

    self.dim1 = checker.check_positive_integer(dim1)
    self.dim2 = checker.check_positive_integer(dim2)
    self.constraint = None
    if max_norm is not None and max_norm > 0:
      self.constraint = tf.keras.constraints.max_norm(max_norm, axis=0)


  @property
  def structure_tail(self):
     activation = ''
     if self._activation is not None:
        activation = '->act'
        if isinstance(self._activation_string, str):
           activation = '->' + self._activation_string
     return '({}x{})'.format(self.dim1, self.dim2) + activation


  def forward(self, x, **kwargs):
    """Implementation Reference:
       [1] https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
    """
    assert isinstance(x, tf.Tensor) and len(x.shape) == 3
    dim1, dim2 = self.dim1, self.dim2
    xd1, xd2 = x.shape.as_list()[1:]

    # TODO: to be capsulate into hyper
    # Get weights
    W1 = tf.get_variable('W1', shape=[dim1, xd1], dtype=th.dtype,
                         initializer=self._weight_initializer,
                         constraint=self.constraint)
    W2 = tf.get_variable('W2', shape=[xd2, dim2], dtype=th.dtype,
                         initializer=self._weight_initializer,
                         constraint=self.constraint)
    # Do bilinear calculation [1]
    XT = tf.matrix_transpose(x)
    XTW1T = tf.matmul(tf.reshape(XT, [-1, xd1]), W1, transpose_b=True)
    W1X = tf.matrix_transpose(tf.reshape(XTW1T, [-1, xd2, dim1]))
    y = tf.reshape(tf.reshape(W1X, [-1, xd2]) @ W2, [-1, dim1, dim2])

    # Add bias if necessary
    if self._use_bias:
      bias = tf.get_variable('bias', shape=[dim2], dtype=th.dtype,
                             initializer=self._bias_initializer)
      y = tf.nn.bias_add(y, bias)

    # Apply activation if provided
    if self._activation: y = self._activation(y)

    return y


