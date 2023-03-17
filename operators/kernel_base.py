from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from tframe import tf

from tframe import context
from tframe import checker
from tframe import hub
from tframe import linker
from tframe import initializers

from typing import Optional


class KernelBase(object):
  """Kernel base provides support for network pruning algorithms via
  KernelBase._get_weights method. Layers or operators using this method
  to allocate trainable variables usually have `hyper` prefix in their name.

  To create sparse mask for pruning:
  (1) set th.prune_on or th.etch_on to True
  (2) initiate the subclass of KernelBase by setting `prune_frac` > 0 or
      `etch` to a string, e.g., lottery:prune_frac=0.1.
  After these 2 steps, the learnable weight of this kernel base will be
  registered to context.pruner, the corresponding etch kernel will be created,
  and corresponding masked weights will replace the trainable weights.
  """

  def __init__(self,
               kernel_key,
               num_units,
               initializer,
               prune_frac=0,
               etch: Optional[str] = None,
               weight_dropout=0.0,
               **kwargs):

    self.kernel_key = checker.check_type(kernel_key, str)
    self.kernel = self._get_kernel(kernel_key)
    self.num_units = checker.check_positive_integer(num_units)

    self.initializer = initializers.get(initializer)
    assert 0 <= prune_frac <= 1
    # IMPORTANT
    self.prune_frac = 0
    if hub.prune_on: self.prune_frac = prune_frac * hub.pruning_rate
    self.etch = etch

    self.weight_dropout = checker.check_type(weight_dropout, float)
    assert 0 <= self.weight_dropout < 1

    self.kwargs = kwargs
    self._check_arguments()

  @property
  def prune_is_on(self):
    # assert 0 <= self.prune_frac <= 1
    # return self.prune_frac > 0 and hub.prune_on
    return hub.prune_on

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
    """This method is crucial for pruning algorithm"""

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

    # Binarize weights if required
    if hub.binarize_weights:
      # See this paper: https://arxiv.org/pdf/1602.02830.pdf
      return self.binarize_weights(weights)

    # If no mask is needed to be created, return weight variable directly
    if not any([self.prune_is_on, self.being_etched, hub.force_to_use_pruner]):
      return weights

    # Register, context.pruner should be created in early model.build
    assert context.pruner is not None
    # Merged lottery logic into etch logic
    if self.prune_is_on:
      assert not self.being_etched
      self.etch = 'lottery:prune_frac={}'.format(self.prune_frac)

    # Register etch kernel to pruner
    masked_weights = context.pruner.register_to_kernels(weights, self.etch)

    # Return
    assert isinstance(masked_weights, tf.Tensor)
    return masked_weights


  # region: BNN related stuff

  @staticmethod
  @tf.custom_gradient
  def binarize_weights(x):
    from tframe import hub as th
    def sign(v):
      return (tf.cast(tf.math.greater_equal(v, 0), th.dtype) - 0.5) * 2
    def grad(dy): return dy
    return sign(x), grad

  # endregion: BNN related stuff



