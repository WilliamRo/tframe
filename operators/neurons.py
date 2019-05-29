from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import collections
import tensorflow as tf

from tframe import checker
from tframe import linker

from .apis.neurobase import NeuroBase
from .apis.generic_neurons import GenericNeurons
from .apis.psi_kernels import PsyKernels


class NeuronArray(NeuroBase, GenericNeurons, PsyKernels):

  class Keys:
    weight_initializer = 'weight_initializer'

  def __init__(
      self,
      scope,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=False,
      bias_initializer='zeros',
      **properties):

    # Call api's initializer
    NeuroBase.__init__(
      self, activation, weight_initializer, use_bias, bias_initializer)

    self.scope = checker.check_type(scope, str)
    self.properties = properties

    self._input_dict = collections.OrderedDict()

  # region : Properties

  # endregion : Properties

  # region : Link

  def __call__(self, dim, *input_list, **input_dict):
    """Link neuron array to graph
    :param output_dim: output dimension
    :param input_list: inputs to be fully connected
    :param input_dict: inputs to be connected with dynamic weights
                       keys are tensors, values correspond to dynamic kernels
    :return: neuron outputs
    """
    # Add inputs
    if input_list: self.add_input(
      tf.concat(input_list, axis=1) if len(input_list) > 1 else input_list[0])
    for x, kernel in input_dict.items(): self.add_input(x, kernel)

    # Make sure input_dict is not empty
    assert self._input_dict

    # Link
    with tf.variable_scope(self.scope):

      # Calculate summed input a, ref: Ba, etc. Layer Normalization, 2016
      a_list = [kernel(dim, x) for x, kernel in self._input_dict.items()]
      a = a_list[0] if len(a_list) == 1 else tf.add_n(a_list, 'summed_inputs')

      # Add bias if necessary
      if self._use_bias: a = self._add_bias(a)
      # Activate if necessary
      if self._activation: a = self._activation(a)

    return a

  # endregion : Link

  # region : Private Methods

  def _psi(self, dim, suffix, *tensors):
    return self.psi_k(dim, *tensors, suffix=suffix,
                      weight_initializer=self._weight_initializer)

  def _add_bias(self, a):
    return self.add_bias(a, self._bias_initializer)

  # endregion : Private Methods

  # region : Public Methods

  def add_input(self, tensor, kernel_id='fc', suffix=None, **kwargs):
    """Add input to neuron array. add_input(x) will add a dense kernel for x.

    :param tensor: input tensor to a psy kernel
    :param kernel_id: kernel identifier, can be a string in psi_kernel
                      directory or a callable method
    :param suffix: suffix of the variable scope of this kernel:
                   scope = 'psy_' + suffix.
    :param kwargs: additional arguments used in kernel function, should be
                   taken care of
    """
    assert isinstance(tensor, tf.Tensor)
    # Set default suffix if not provided
    if not suffix: suffix = str(len(self._input_dict) + 1)

    # Check kernel arguments
    if callable(kernel_id):
      args = inspect.getfullargspec(kernel_id).args
      if len(args) != 2: raise AssertionError(
        '!! Illegal argument specification `{}`. Callable kernel provided '
        'should have exactly 2 arguments.'.format(args))
    else:
      assert kernel_id in self.directory
      args = inspect.getfullargspec(self.directory[kernel_id]).args
      for arg_name in args[2:]:
        if not arg_name in kwargs:
          # Use default weight initializer if not provided
          if arg_name == self.Keys.weight_initializer:
            kwargs[self.Keys.weight_initializer] = self._weight_initializer
          else: raise AssertionError(
            '!! kernel ({}) argument `{}` should be provided.'.format(
              kernel_id, arg_name))
      # Make sure provided arguments matches kernel argument specification
      # .. exactly
      if len(kwargs) != len(args[2:]):
        raise AssertionError(
          '!! kernel `{}` requires {} additional arguments but {} are '
          'provided.'.format(kernel_id, len(args[2:]), len(kwargs)))

    # Add input to dict
    self._input_dict[tensor] = self.get_kernel(kernel_id, suffix, **kwargs)

  # endregion : Public Methods

  # region : Static Methods

  @staticmethod
  def inherit(nb, scope, activation=None, is_gate=False, **kwargs):
    """Returns a NeuronArray whose NeuroBase attribute inherits from another
      instance of NeuroBase"""
    assert isinstance(nb, NeuroBase)
    if activation is None and is_gate: activation = tf.sigmoid
    return NeuronArray(scope, activation, nb._weight_initializer,
                       nb._use_bias, nb._bias_initializer, **kwargs)

  # endregion : Static Methods

  # region : Library

  def mul_neuro_11(self, x, s, fd, scope, activation=None,
                   hyper_initializer=None):
    na = NeuronArray(scope, activation, self._weight_initializer,
                     self._use_bias, self._bias_initializer)
    na.add_input(x, suffix='x')
    if hyper_initializer is None: hyper_initializer = self._weight_initializer
    na.add_input(s, kernel_id='mul', suffix='s', seed=x, fd=fd,
                 weight_initializer=hyper_initializer)
    return na(linker.get_dimension(s))

  # endregion : Library

