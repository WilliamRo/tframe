from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import linker
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.distributor import Distributor
from tframe.operators.neurons import NeuronArray


class MGDU(CellBase, Distributor, NeuronArray):

  net_name = 'mgdu'

  def __init__(
      self,
      configs,
      factoring_dim=None,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      reverse=False,
      **kwargs):

    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer,
                      use_bias, bias_initializer, **kwargs)

    # Specific attributes
    self._reverse = checker.check_type(reverse, bool)
    self._groups = self._get_groups(configs)
    self._state_size = self._get_total_size(self._groups)

    if factoring_dim is None: factoring_dim = self._state_size
    self._fd = checker.check_positive_integer(factoring_dim)


  @property
  def _scale_tail(self):
    config_str = self._get_config_string(self._groups, reverse=self._reverse)
    tail = '({}|fd{})'.format(config_str, self._fd)
    return tail


  def _get_sog_activation(self, x, s, configs, scope, name):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    net_u = self.mul_neuro_11(x, s, self._fd, scope)
    return self._softmax_over_groups(net_u, configs, name)


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate update gates
    u, z = self._get_coupled_gates(
      x, prev_s, self._groups, reverse=self._reverse)
    self._gate_dict['beta_gate'] = z

    # - Calculate s_bar
    s_bar = self.mul_neuro_11(x, prev_s, self._fd, 's_bar', self._activation)

    # - Update state
    with tf.name_scope('transit'):
      new_s = tf.add(tf.multiply(z, prev_s), tf.multiply(u, s_bar))

    # - Calculate output and return
    y = new_s
    return y, new_s


