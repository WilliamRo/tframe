from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

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
      psi_config=None,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      reverse=False,
      **kwargs):
    """
    :param psi_config: e.g. 's:xs+g;xs', 's:x'
    """

    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer,
                      use_bias, bias_initializer, **kwargs)

    # Specific attributes
    self._reverse = checker.check_type(reverse, bool)
    self._groups = self._get_groups(configs)
    self._state_size = self._get_total_size(self._groups)

    if factoring_dim is None: factoring_dim = self._state_size
    self._fd = checker.check_positive_integer(factoring_dim)

    if not psi_config: psi_config = 's:x'
    self._psi_string = checker.check_type(psi_config, str)
    self._psi_config = self._parse_psi_string()


  @property
  def _scale_tail(self):
    config_str = self._get_config_string(self._groups, reverse=self._reverse)
    tail = '({}|fd{}|{})'.format(config_str, self._fd, self._psi_string)
    return tail


  def _get_sog_activation(self, x, s, configs, scope, name):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    net_u = self._generic_neurons(x, s, self._psi_config['g'], scope)
    return self._softmax_over_groups(net_u, configs, name)


  def _generic_neurons(self, x, s, config, scope, activation=None):
    if not config: return self.neurons(x, s, scope=scope, activation=activation)
    # Prepare seed
    seed_list = []
    if 'x' in config: seed_list.append(x)
    if 's'in config: seed_list.append(s)
    seed = seed_list[0] if len(seed_list) == 1 else tf.concat(seed_list, axis=1)
    return self.mul_neuro_11(x, s, self._fd, scope, activation, seed)


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate update gates
    u, z = self._get_coupled_gates(
      x, prev_s, self._groups, reverse=self._reverse)
    self._gate_dict['beta_gate'] = z

    # - Calculate s_bar
    s_bar = self._generic_neurons(
      x, prev_s, self._psi_config['s'], 's_bar', self._activation)

    # - Update state
    with tf.name_scope('transit'):
      new_s = tf.add(tf.multiply(z, prev_s), tf.multiply(u, s_bar))

    # - Calculate output and return
    y = new_s
    return y, new_s


  def _parse_psi_string(self):
    config = {'g': None, 's': None}
    sub_strs = self._psi_string.split('+')
    assert 0 < len(sub_strs) <= 2
    for sub_str in sub_strs:
      assert isinstance(sub_str, str)
      cfg = sub_str.split(':')
      assert len(cfg) == 2 and cfg[0] in ('s', 'g')
      self._check_psi_config(cfg[1])
      config[cfg[0]] = cfg[1]
    return config


  @staticmethod
  def _check_psi_config(config):
    assert isinstance(config, str) and 0 < len(config) <= 2
    for token in config: assert token in ('x', 's')


