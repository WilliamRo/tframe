from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tensorflow.python.ops import init_ops

import tframe as tfr
from tframe.operators.apis.neurobase import NeuroBase

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
               moving_var_initializer=init_ops.ones_initializer(),
               use_tf_bn=False):
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

    self._use_tf_bn = use_tf_bn

    # Apply global momentum and epsilon if necessary
    if tfr.hub.bn_momentum is not None:
      self.momentum = tfr.hub.bn_momentum
    if tfr.hub.bn_epsilon is not None:
      self.epsilon = tfr.hub.bn_epsilon


  @single_input
  def _link(self, input_:tf.Tensor, **kwargs):
    """Since fullname is defined, linking will be done inside a variable 
       scope"""

    # Get is_training placeholder
    assert len(tf.get_collection(pedia.is_training)) == 1
    is_training = tf.get_collection(pedia.is_training)[0]

    # Use tensorflow batch normalization layer if required
    if self._use_tf_bn: return tf.layers.batch_normalization(
      input_, axis=self.axis, momentum=self.momentum, epsilon=self.epsilon,
      center=self.center, scale=self.scale,
      beta_initializer=self.beta_initializer,
      gamma_initializer=self.gamma_initializer,
      moving_mean_initializer=self.moving_mean_initializer,
      moving_variance_initializer=self.moving_var_initializer,
      training=is_training, reuse=self.linked)

    # region : Get input shape and validation check

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

    # Get parameters shape (only support most common use-case currently)
    if len(self.axis) != 1:
      raise ValueError('!! Single axis batch norm is supported only currently')
    param_shape = (input_shape[self.axis[0]].value,)

    # Calculate mean and variance
    reduction_axes = [i for i in range(len(input_shape))
                      if i not in self.axis]
    batch_mean, batch_variance = tf.nn.moments(input_, reduction_axes)

    # If this layer has been linked before, reuse the variables
    if self.linked: tf.get_variable_scope().reuse_variables()

    def _get_variable(name, initializer, trainable=True):
      return tf.get_variable(
        name, param_shape, tfr.hub.dtype, initializer, trainable=trainable)

    # Create un-trainable variables for moving mean and var
    moving_mean = _get_variable(
      'moving_mean', self.moving_mean_initializer, False)
    moving_var = _get_variable(
      'moving_var', self.moving_var_initializer, False)

    # Create update op for moving_(mean|var)
    update_moving_average = lambda ma, v: tf.assign(
      ma, self.momentum * ma + (1 - self.momentum) * v)

    # DO NOT CREATE 'update_ops' HERE in tensorflow 1.14.0

    def mean_var_with_update():
      # IMPORTANT NOTE:
      #   update_ops should be created inside tf.cond, otherwise this op will
      #   be executed unconditionally !!!!!!!
      update_ops = [update_moving_average(moving_mean, batch_mean),
                    update_moving_average(moving_var, batch_variance)]
      sentry = tf.assert_equal(
        is_training, True, message='Update moving average while not training')
      with tf.control_dependencies(update_ops + [sentry]):
        return tf.identity(batch_mean), tf.identity(batch_variance)

    # Get mean variance according to is_training
    mean, variance = tf.cond(
      is_training, mean_var_with_update, lambda: (moving_mean, moving_var))

    # Get variable
    if self.center: self.beta = _get_variable('beta', self.beta_initializer)

    if self.scale: self.gamma = _get_variable('gamma', self.gamma_initializer)

    # Output
    output = tf.nn.batch_normalization(
      input_, mean, variance, self.beta, self.gamma, self.epsilon)

    return output


class LayerNormalization(Layer):

  full_name = 'layernorm'
  abbreviation = 'ln'

  def __init__(self,
               axis=-1,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               epsilon=1e-3):
    self.axis = axis
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.epsilon = epsilon

  @single_input
  def _link(self, x):
    return NeuroBase.layer_normalize(
      x=x,
      axis=self.axis,
      center=self.center,
      scale=self.scale,
      beta_initializer=self.beta_initializer,
      gamma_initializer=self.gamma_initializer,
      epsilon=self.epsilon)















