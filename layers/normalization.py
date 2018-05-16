from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.layers import utils

from .layer import Layer
from .layer import single_input

from ..utils import tensor_shape

from .. import initializers
from .. import pedia


class BatchNormalization(Layer):
  """Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"
  """

  full_name = 'batchnorm'
  abbreviation = 'bn'

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=init_ops.zeros_initializer(),
               gamma_initializer=init_ops.ones_initializer(),
               moving_mean_initializer=init_ops.zeros_initializer(),
               moving_var_initializer=init_ops.ones_initializer()):
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_var_initializer = initializers.get(moving_var_initializer)

    self.beta = None
    self.gamma = None

  @single_input
  def _link(self, input_, **kwargs):
    """Since fullname is defined, linking will be done inside a variable 
       scope"""
    # region : Get input shape and validation check

    assert isinstance(input_, tf.Tensor)
    input_shape = input_.get_shape().as_list()
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('!! Input has undefined rank: {}'.format(input_shape))
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]
    if not isinstance(self.axis, list):
      raise TypeError('!! Axis must be int or list, type given: {}'.format(
        type(self.axis)))
    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axis
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('!! Invalid axis: {}'.format(x))
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('!! Duplicate axis: {}'.format(self.axis))

    # endregion : Get input shape and validation check

    is_training = tf.get_collection(pedia.is_training)[0]

    # Get parameters shape (only support most common use-case currently)
    if len(self.axis) != 1:
      raise ValueError('!! Single axis batch norm is supported only currently')
    param_shape = (input_shape[self.axis[0]].value,)

    # Calculate mean and variance
    reduction_axes = [i for i in range(len(input_shape))
                      if i not in self.axis]
    batch_mean, batch_variance = tf.nn.moments(input_, reduction_axes)

    # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(decay=self.momentum)

    # Create the shadow variables and add an ops to maintain moving averages
    # for mean and variance
    ema_apply_op = ema.apply([batch_mean, batch_variance])
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_variance)

    mean, variance = tf.cond(
      is_training, mean_var_with_update,
      lambda: (ema.average(batch_mean), ema.average(batch_variance)))

    # If this layer has been linked before, reuse the variables
    if self.center and self.beta is not None or (
          self.scale and self.gamma is not None):
      tf.get_variable_scope().reuse_variables()

    # Get variable
    if self.center:
      self.beta = tf.get_variable(
        'beta', shape=param_shape, dtype=tf.float32,
        initializer=self.beta_initializer , trainable=True)

    if self.scale:
      self.gamma = tf.get_variable(
        'gamma', shape=param_shape, dtype=tf.float32,
        initializer=self.gamma_initializer, trainable=True)

    # Output
    output = tf.nn.batch_normalization(
      input_, mean, variance, self.beta, self.gamma, self.epsilon)

    return output


















