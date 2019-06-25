from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import activations
from tframe import checker
from tframe import hub
from tframe import initializers
from tframe import linker


class NeuroBase(object):

  def __init__(
      self,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=False,
      bias_initializer='zeros',
      layer_normalization=False,
      weight_dropout=0.0,
      **kwargs):

    if activation: activation = activations.get(activation)
    self._activation = activation
    self._weight_initializer = initializers.get(weight_initializer)
    self._use_bias = checker.check_type(use_bias, bool)
    self._bias_initializer = initializers.get(bias_initializer)
    self._layer_normalization = checker.check_type(layer_normalization, bool)
    self._gain_initializer = initializers.get(
      kwargs.get('gain_initializer', 'ones'))
    self._normalize_each_psi = kwargs.pop('normalize_each_psi', False)
    self._weight_dropout = checker.check_type(weight_dropout, float)
    assert 0 <= self._weight_dropout < 1
    self._nb_kwargs = kwargs

  @property
  def _prune_frac(self):
    """Weight prune fraction in lottery finding"""
    return self._nb_kwargs.get('prune_frac', 0)

  @property
  def lottery_activated(self):
    """This property is decided in linker.neuron"""
    return hub.prune_on and self._prune_frac > 0

  # region : Public Methods

  def differentiate(
      self, num_neurons, name, activation=None,
      weight_initializer=None, use_bias=None, bias_initializer=None,
      layer_normalization=None, normalize_each_psi=None,
      weight_dropout=None, **kwargs):
    from tframe.operators.neurons import NeuronArray
    """A neuron group can differentiate to produce a neuron array which shares
      part of attributes in NeuroBase.
      
      A more elegant way is letting NeuroBase and NeuronArray inheriting from
      a same parent class. But the developer does not want to do this for now.
      
      TODO: this is a bad designed and should be refactored to be more elegant
    """
    if weight_initializer is None: weight_initializer = self._weight_initializer
    if use_bias is None: use_bias = self._use_bias
    if bias_initializer is None: bias_initializer = self._bias_initializer
    if layer_normalization is None:
      layer_normalization = self._layer_normalization
    if normalize_each_psi is None: normalize_each_psi = self._normalize_each_psi
    if weight_dropout is None: weight_dropout = self._weight_dropout

    merged_dict = {}
    merged_dict.update(self._nb_kwargs)
    merged_dict.update(kwargs)

    return NeuronArray(
      num_neurons, name, activation=activation,
      weight_initializer=weight_initializer,
      use_bias=use_bias, bias_initializer=bias_initializer,
      layer_normalization=layer_normalization,
      normalize_each_psi=normalize_each_psi,
      weight_dropout=weight_dropout,
      **merged_dict)

  @staticmethod
  def get_dimension(x):
    """Get tensor dimension
    :param x: a tensor of shape [batch_size, dimension]
    :return: dimension of x
    """
    assert isinstance(x, tf.Tensor) and len(x.shape) == 2
    return x.shape.as_list()[-1]

  @staticmethod
  def dropout(input_, dropout_rate):
    return linker.dropout(input_, dropout_rate)

  # endregion : Public Methods

  # region : Library

  def dense_v2(self, num_neurons, scope, *inputs, activation=None,
               num_or_size_splits=None, is_gate=False, **kwargs):
    # Sanity check
    assert isinstance(inputs, (tuple, list))
    if activation is None and is_gate: activation = tf.sigmoid
    na = self.differentiate(num_neurons, scope, activation, **kwargs)
    output = na(*inputs)
    # Split if necessary
    if num_or_size_splits is not None:
      return tf.split(output, num_or_size_splits, axis=1)
    return output

  def dense(self, output_dim, x, scope, activation=None,
            num_or_size_splits=None, **kwargs):
    """Dense neuron.
    :param output_dim: neuron number (output dimension)
    :param x: input tensor of shape [batch_size, input_dim]
    :param scope: scope
    :param activation: activation function
    :param num_or_size_splits: if provided, tf.split will be used to output
    :param kwargs: other setting which will be flow into neuron array
                   and finally caught by NeuroBase._nb_kwargs
    :return: a tensor of shape [batch_size, output_dim] if split option if off
    """
    na = self.differentiate(output_dim, scope, activation, **kwargs)
    if not self.lottery_activated: output = na(x)
    else:
      assert not na.being_etched
      na.add_kernel(x, suffix='x', prune_frac=self._prune_frac)
      output = na()
    # Split if necessary
    if num_or_size_splits is not None:
      return tf.split(output, num_or_size_splits, axis=1)
    return output

  # endregion : Library


class RNeuroBase(NeuroBase):

  def __init__(
      self,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=False,
      bias_initializer='zeros',
      layer_normalization=False,
      dropout_rate=0.0,
      zoneout_rate=0.0,
      weight_dropout=0.0,
      **kwargs):

    super().__init__(
      activation=activation,
      weight_initializer=weight_initializer,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      layer_normalization=layer_normalization,
      weight_dropout=weight_dropout,
      **kwargs)

    self._dropout_rate = checker.check_type(dropout_rate, (int, float))
    assert 0 <= dropout_rate < 1
    self._zoneout_rate = checker.check_type(zoneout_rate, (int, float))
    assert 0 <= zoneout_rate < 1

  @property
  def _s_prune_frac(self):
    """State weight matrix pruning fraction during lottery finding"""
    frac = self._nb_kwargs.get('s_prune_frac', 0)
    if frac == 0: return self._prune_frac
    else: return frac

  @property
  def _x_prune_frac(self):
    """Pruning fraction of input weight matrix during lottery finding"""
    frac =  self._nb_kwargs.get('x_prune_frac', 0)
    if frac == 0: return self._prune_frac
    else: return frac

  @property
  def lottery_activated(self):
    """Whether to find lottery in this recurrent neuron cluster"""
    return hub.prune_on and (self._s_prune_frac > 0 or self._x_prune_frac > 0)

  # region : Public Methods

  def differentiate(
      self, num_neurons, name, activation=None, weight_initializer=None,
      use_bias=None, bias_initializer=None, layer_normalization=None,
      normalize_each_psi=None, weight_dropout=None, is_gate=False, **kwargs):
    """A cell can differentiate to produce a neuron array which shares
      part of attributes in NeuroBase"""

    if activation is None and is_gate: activation = tf.sigmoid
    return super().differentiate(
      num_neurons, name, activation, weight_initializer, use_bias,
      bias_initializer, layer_normalization, normalize_each_psi,
      weight_dropout, **kwargs)

  @staticmethod
  def get_state_size(s):
    return NeuroBase.get_dimension(s)

  # endregion : Public Methods

  # region : Library

  def dense_rn(self, x, s, scope, activation=None, output_dim=None,
               is_gate=False, num_or_size_splits=None, **kwargs):
    """Dense recurrent neuron"""
    if output_dim is None: output_dim = self.get_state_size(s)
    na = self.differentiate(
      output_dim, scope, activation, is_gate=is_gate, **kwargs)
    # If don't need to prune
    if not self.lottery_activated: output = na(x, s)
    else:
      assert not na.being_etched
      # Add x and s separately
      na.add_kernel(x, suffix='x', prune_frac=self._x_prune_frac)
      na.add_kernel(s, suffix='s', prune_frac=self._s_prune_frac)
      output = na()
    # Split if necessary
    if num_or_size_splits is not None:
      return tf.split(output, num_or_size_splits, axis=1)
    return output

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

  def reset_14(self, x, s, scope, activation, output_dim=None, reset_s=True,
               return_gate=False):
    """Force reset_s option to be True for now. reset_s=False corresponds to
       .. another variants
    """
    if output_dim is None: output_dim = linker.get_dimension(s)
    state_size = linker.get_dimension(s)
    # Calculate the reset gate
    gate_dim = state_size if reset_s else output_dim
    r = self.dense_rn(x, s, 'reset_gate', output_dim=gate_dim, is_gate=True)

    # Calculate s_bar
    if reset_s:
      y = self.dense_rn(x, r * s, scope, activation, output_dim)
    else: raise NotImplementedError

    # Return
    if return_gate: return y, r
    else: return y

  # endregion : Library

