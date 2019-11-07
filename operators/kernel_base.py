from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import tensorflow as tf

from tframe import context
from tframe import checker
from tframe import hub
from tframe import linker
from tframe import initializers
from tframe import regularizers

from tframe.operators.prune.etches import get_etch_kernel


class KernelBase(object):

  def __init__(self,
               kernel_key,
               num_neurons,
               initializer,
               prune_frac=0,
               etch=None,
               weight_dropout=0.0,
               **kwargs):

    self.kernel_key = checker.check_type(kernel_key, str)
    self.kernel = self._get_kernel(kernel_key)
    self.num_neurons = checker.check_positive_integer(num_neurons)

    self.initializer = initializers.get(initializer)
    assert 0 <= prune_frac <= 1
    # IMPORTANT
    self.prune_frac = prune_frac * hub.pruning_rate_fc
    self.etch = etch

    self.weight_dropout = checker.check_type(weight_dropout, float)
    assert 0 <= self.weight_dropout < 1

    self.kwargs = kwargs
    self._check_arguments()

  @property
  def prune_is_on(self):
    assert 0 <= self.prune_frac <= 1
    return self.prune_frac > 0 and hub.prune_on

  @property
  def being_etched(self):
    return self.etch is not None and hub.etch_on


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
    # Get regularizer if necessary
    regularizer = None
    if hub.use_global_regularizer: regularizer = hub.get_global_regularizer()
    # Get constraint if necessary
    constraint = hub.get_global_constraint()
    # Get weights
    weights = tf.get_variable(name, shape, dtype=dtype, initializer=initializer,
                              regularizer=regularizer, constraint=constraint)
    # If weight dropout is positive, dropout and return
    if self.weight_dropout > 0:
      return linker.dropout(weights, self.weight_dropout, rescale=True)
    # If no mask is needed to be created, return weight variable directly
    if not any([self.prune_is_on, self.being_etched]): return weights
    # Register, context.pruner should be created in early model.build
    assert context.pruner is not None
    # Merged lottery logic into etch logic
    if self.prune_is_on:
      assert not self.being_etched
      self.etch = 'lottery:prune_frac={}'.format(self.prune_frac)

    # Register etch kernel to pruner
    masked_weights = context.pruner.register_to_dense(weights, self.etch)

    # if self.prune_is_on:
    #   masked_weights = context.pruner.register_to_dense(
    #     weights, self.prune_frac)
    # else:
    #   # TODO
    #   assert self.being_etched
    #   mask = self._get_etched_surface(weights)
    #   masked_weights = context.pruner.register_with_mask(weights, mask)

    # Return
    assert isinstance(masked_weights, tf.Tensor)
    return masked_weights


  # def _get_etched_surface(self, weights):
  #   assert isinstance(self.etch, str) and isinstance(weights, tf.Variable)
  #   mask = tf.get_variable(
  #     'etched_surface', shape=weights.shape, dtype=hub.dtype,
  #     initializer=tf.initializers.ones)
  #   # Get etch kernel and register to pruner
  #   kernel = get_etch_kernel(self.etch)
  #   etch_kernel = kernel(weights, mask)
  #   context.pruner.register_etch_kernel(etch_kernel)
  #   return mask
