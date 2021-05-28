from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import linker
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.distributor import Distributor
from tframe.operators.neurons import NeuronArray
from tframe.operators.apis.hyper_kernel import HyperKernel


class HyperGDU(CellBase, Distributor, NeuronArray, HyperKernel):
  """Hyper Grouped Distributor Unit
     Reference:
       [1] Grouped Distributor Unit, 2019
       [2] Hyper Networks, 2016"""
  net_name = 'gdu'

  def __init__(
      self,
      configs,
      hyper_kernel,
      hyper_dim,
      signal_size,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      reverse=False,
      use_reset_gate=False,
      **kwargs):
    """
    :param configs: a list or tuple of tuples with format (size, num, delta)
                    or a string with format `S1xM1xD1+S2xM2xD2+...`
    """
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer, use_bias,
                      bias_initializer, **kwargs)
    # NeuronArray's constructor will not be called (which is not elegant)

    # Specific attributes
    self._reverse = checker.check_type(reverse, bool)
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)

    self._groups = self._get_groups(configs)
    self._state_size = self._get_total_size(self._groups)

    self.kernel_key = checker.check_type(hyper_kernel, str)
    self._hyper_kernel = self._get_hyper_kernel(hyper_kernel)
    self._hyper_dim = checker.check_positive_integer(hyper_dim)
    self._signal_size = checker.check_positive_integer(signal_size)

    # Assigned during linking
    self._gate_signals = None

    assert not self._use_reset_gate
    
    
  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = (
      self._get_placeholder('s', self._state_size),
      self._get_hyper_state_holder(self.kernel_key, self._hyper_dim))
    return self._init_state


  @property
  def _scale_tail(self):
    config_str = self._get_config_string(self._groups, reverse=self._reverse)
    tail = '({}>{}>{})'.format(self._hyper_dim, self._signal_size, config_str)
    if self._use_reset_gate: tail = '[r]' + tail
    return tail


  def _get_sog_activation(self, x, s, configs, scope, name):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    assert isinstance(self._gate_signals, list) and len(self._gate_signals) == 3
    net_u = self._hyper_neuron_16(x, s, self._gate_signals, scope)
    u = linker.softmax_over_groups(net_u, configs, name)
    return u


  def _link(self, prev_s, x, **kwargs):
    s, s_hat = prev_s
    self._check_state(s)

    # - Get hyper output (new_s_hat)
    # .. hyper input
    x_hat = tf.concat([s, x], axis=1, name='x_hat')
    seed, new_s_hat = self._hyper_kernel(x_hat, s_hat)

    # - Get signals (state signals and gate signals)
    s_signals, g_signals = self._get_embeddings_hyper16(
      seed, num_cluster=2, signal_size=self._signal_size)
    self._gate_signals = g_signals

    # - Calculate update gates
    u, z = self._get_coupled_gates(x, s, self._groups, reverse=self._reverse)
    self._gate_dict['beta_gate'] = z

    # - Calculate s_bar
    s_bar = self._hyper_neuron_16(x, s, s_signals, 's_bar', self._activation)

    # - Update state
    with tf.name_scope('transit'):
      new_s = z * s + u * s_bar

    # - Calculate output and return
    y = new_s
    return y, (new_s, new_s_hat)

