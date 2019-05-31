from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import linker
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.distributor import Distributor
from tframe.operators.neurons import NeuronArray


class GDU(CellBase, Distributor, NeuronArray):
  """Grouped Distributor Unit
     Reference: gdu2019"""
  net_name = 'gdu'

  def __init__(
      self,
      configs,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      reverse=False,
      use_reset_gate=False,
      layer_normalization=False,
      **kwargs):
    """
    :param configs: a list or tuple of tuples with format (size, num, delta)
                    or a string with format `S1xM1xD1+S2xM2xD2+...`
    """
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer, use_bias,
                      bias_initializer, layer_normalization, **kwargs)

    # Specific attributes
    self._reverse = checker.check_type(reverse, bool)
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)

    self._groups = self._get_groups(configs)
    self._state_size = self._get_total_size(self._groups)


  @property
  def _scale_tail(self):
    config_str = self._get_config_string(self._groups, reverse=self._reverse)
    tail = '({})'.format(config_str)
    if self._use_reset_gate: tail = '[r]' + tail
    return tail


  def _get_sog_activation(self, x, s, configs, scope, name):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    net_u = self.dense_rn(x, s, scope)
    u = linker.softmax_over_groups(net_u, configs, name)
    return u


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate update gates
    u, z = self._get_coupled_gates(
      x, prev_s, self._groups, reverse=self._reverse)
    self._gate_dict['beta_gate'] = z

    # - Calculate s_bar
    if self._use_reset_gate:
      s_bar, r = self.reset_14(
        x, prev_s, 's_bar', self._activation, reset_s=True, return_gate=True)
      self._gate_dict['reset_gate'] = r
    else:
      s_bar = self.dense_rn(x, prev_s, 's_bar', self._activation)

    # - Update state
    with tf.name_scope('transit'):
      new_s = z * prev_s + u * s_bar

    # - Calculate output and return
    y = new_s
    return y, new_s

