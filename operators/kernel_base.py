from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import tensorflow as tf

from tframe import context
from tframe import checker
from tframe import hub
from tframe import initializers


class KernelBase(object):

  def __init__(self,
               kernel_key,
               num_neurons,
               initializer,
               prune_frac=0,
               **kwargs):

    self.kernel_key = checker.check_type(kernel_key, str)
    self.kernel = self._get_kernel(kernel_key)
    self.num_neurons = checker.check_positive_integer(num_neurons)

    self.initializer = initializers.get(initializer)
    assert 0 <= prune_frac <= 1
    self.prune_frac = prune_frac

    self.kwargs = kwargs
    self._check_arguments()


  @property
  def prune_is_on(self):
    assert 0 <= self.prune_frac <= 1
    return self.prune_frac > 0 and hub.prune_on


  def __call__(self): raise NotImplementedError


  def _get_kernel(self, kernel_key): raise NotImplementedError


  def _check_arguments(self):
    # The 1st argument is self
    arg_names = inspect.getfullargspec(self.kernel).args[1:]
    for arg_name in arg_names:
      if not arg_name in self.kwargs: raise AssertionError(
        '!! kernel ({}) argument `{}` should be provided.'.format(
          self.kernel_key, arg_name))

    # Make sure provided arguments matches kernel argument specification
    # .. exactly
    if len(self.kwargs) != len(arg_names):
      raise AssertionError(
        '!! kernel `{}` requires {} additional arguments but {} are '
        'provided.'.format(self.kernel_key, len(arg_names), len(self.kwargs)))


  def _get_weights(self, name, shape, dtype=None, initializer=None):
    if initializer is None: initializer = self.initializer
    else: initializer = initializers.get(initializer)
    # Set default dtype if not specified
    if dtype is None: dtype = hub.dtype
    # Get weights
    weights = tf.get_variable(name, shape, dtype=dtype, initializer=initializer)
    if not self.prune_is_on: return weights
    # Register, context.pruner should be created in early model.build
    assert context.pruner is not None
    masked_weights = context.pruner.register_to_dense(weights, self.prune_frac)
    # Return
    assert isinstance(masked_weights, tf.Tensor)
    return masked_weights

