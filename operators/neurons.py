from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import hub
from tframe import linker

from .psi_kernel import PsyKernel
from .apis.neurobase import NeuroBase


class NeuronArray(NeuroBase):

  # class Keys: TODO
  #   weight_initializer = 'weight_initializer'

  def __init__(
      self,
      num_neurons,
      scope,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=False,
      bias_initializer='zeros',
      layer_normalization=False,
      **kwargs):

    # Call api's initializer
    NeuroBase.__init__(self, activation, weight_initializer, use_bias,
                       bias_initializer, layer_normalization, **kwargs)

    self.num_neurons = checker.check_positive_integer(num_neurons)
    self.scope = checker.check_type(scope, str)
    self.psi_kernels = []

  # region : Properties

  # endregion : Properties

  # region : Link

  def __call__(self, *input_list):
    """Link neuron array to graph
    :param input_list: inputs to be fully connected
    :return: neuron outputs
    """
    # Add inputs
    if input_list:
      checker.check_type(input_list, tf.Tensor)
      # Concatenation is forbidden when LN is on
      if all([len(input_list) > 1,
              self._layer_normalization, self._normalize_each_psi]):
        for x in input_list: self.add_kernel(x)
      else: self.add_kernel(
        tf.concat(input_list, axis=1) if len(input_list) > 1 else input_list[0])

    # Make sure psi_kernel is not empty
    assert self.psi_kernels

    # Link
    with tf.variable_scope(self.scope):

      # Calculate summed input a, ref: Ba, etc. Layer Normalization, 2016
      a_list = [kernel() for kernel in self.psi_kernels]
      a = a_list[0] if len(a_list) == 1 else tf.add_n(a_list, 'summed_inputs')

      # Do layer normalization here if necessary
      if self._layer_normalization:
        if not self._normalize_each_psi:
          a = PsyKernel.layer_normalization(a, self._gain_initializer, False)
        # If LN is on, use_bias option must be True
        self._use_bias = True

      # Add bias if necessary
      if self._use_bias: a = self._add_bias(a)

      # Activate if necessary
      if self._activation: a = self._activation(a)

    return a

  # endregion : Link

  # region : Private Methods

  def _add_bias(self, a):
    bias = tf.get_variable('bias', shape=[self.num_neurons], dtype=hub.dtype,
                           initializer=self._bias_initializer)
    return tf.nn.bias_add(a, bias)

  # endregion : Private Methods

  # region : Public Methods

  def add_kernel(
      self, input_, kernel_key='fc', suffix=None, weight_initializer=None,
      prune_frac=0, **kwargs):
    """Add a psi kernel to self. Final neuron activation will be
       y = \phi(\Sigma_k(psi_k()) + bias)
    :param input_: psi input
    :param kernel_key: a string indicating which kernel should be called
                       during linking
    :param suffix: suffix of the variable scope of this kernel:
                   scope = 'psy_' + suffix.
    :param weight_initializer: if not provided, will be set to self's
    :param prune_frac: if positive, weights got inside kernel will be masked
                       and may be pruned in `lottery` style
    :param kwargs: additional arguments to call kernel, will be checked during
                   PsiKernel instantiating
    """

    # Set default suffix/weight_initializer if not provided
    if not suffix: suffix = str(len(self.psi_kernels) + 1)
    if weight_initializer is None: weight_initializer = self._weight_initializer

    # Initiate a psi_kernel
    psi_kernel = PsyKernel(
      kernel_key, self.num_neurons, input_, suffix,
      weight_initializer=weight_initializer, prune_frac=prune_frac,
      LN=self._layer_normalization and self._normalize_each_psi,
      gain_initializer=self._gain_initializer, **kwargs)

    self.psi_kernels.append(psi_kernel)

  def differentiate(self, num_neurons, name, activation=None, is_gate=False,
                    **kwargs):
    """Neuron layers or cells can differentiate to produce a sub neuron
       group which shares part of attributes in NeuroBase"""
    if activation is None and is_gate: activation = tf.sigmoid
    return NeuronArray(
      num_neurons, name, activation=activation,
      weight_initializer=self._weight_initializer,
      use_bias=self._use_bias, bias_initializer=self._bias_initializer,
      layer_normalization=self._layer_normalization,
      normalize_each_psi=self._normalize_each_psi, **kwargs)

  # endregion : Public Methods

  # region : Static Methods

  # endregion : Static Methods

  # region : Library

  def dense(self, output_dim, x, scope, activation=None):
    """Dense neuron"""
    na = self.differentiate(output_dim, scope, activation)
    if self._prune_frac == 0: return na(x)
    else:
      na.add_kernel(x, suffix='x', prune_frac=self._prune_frac)
      return na()


  def dense_rn(self, x, s, scope, activation=None, num=None, is_gate=False):
    """Dense recurrent neuron"""
    if num is None: num = linker.get_dimension(s)
    na = self.differentiate(num, scope, activation, is_gate)
    # If don't need to prune
    if self._s_prune_frac == self._x_prune_frac == 0 or not hub.prune_on:
      return na(x, s)
    # Add x
    na.add_kernel(x, suffix='x', prune_frac=self._x_prune_frac)
    # Add s
    na.add_kernel(s, suffix='s', prune_frac=self._s_prune_frac)
    return na()


  def mul_neuro_11(self, x, s, fd, scope, activation=None, seed=None,
                   hyper_initializer=None):
    state_size = linker.get_dimension(s)
    if seed is None: seed = x
    na = self.differentiate(state_size, scope, activation)
    na.add_kernel(x, suffix='x')
    if hyper_initializer is None: hyper_initializer = self._weight_initializer
    na.add_kernel(s, kernel_key='mul', suffix='s', seed=seed, fd=fd,
                  weight_initializer=hyper_initializer)
    return na()


  def reset_14(self, x, s, scope, activation, output_dim=None, reset_s=True):
    """Force reset_s option to be True for now. reset_s=False corresponds to
       .. another variants
    """
    if output_dim is None: output_dim = linker.get_dimension(s)
    state_size = linker.get_dimension(s)
    # Calculate the reset gate
    gate_dim = state_size if reset_s else output_dim
    reset_gate = self.dense_rn(x, s, 'reset_gate', num=gate_dim, is_gate=True)

    # Add reset gate to dict if necessary
    from tframe.nets.rnn_cells.cell_base import CellBase
    if isinstance(self, CellBase): self._gate_dict['reset_gate'] = reset_gate

    # Calculate s_bar
    if reset_s: return self.dense_rn(x, reset_gate * s, scope, activation,
                                     output_dim)
    else: raise NotImplementedError

  # endregion : Library

