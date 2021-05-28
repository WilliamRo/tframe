from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.mixer import Mixer


class Tuner(CellBase, Mixer):

  net_name = 'tuner'

  def __init__(
      self,
      configs,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      use_reset_gate=False,
      **kwargs):

    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer, use_bias,
                      bias_initializer, **kwargs)
    Mixer.__init__(self, configs)

    # Specific attributes
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)

    self._state_size = self.total_size


  @property
  def _scale_tail(self):
    tail = '({})'.format(self.group_string)
    if self._use_reset_gate: tail += '[r]'
    return tail


  def _get_coupled_gates(self, x, s):
    a_dim = self._state_size + sum(
      [n if s > 1 else 0 for s, n in self.groups])
    a = self.dense_rn(x, s, 'net_a', output_dim=a_dim)
    z = self._cumax_over_groups(a)
    self._gate_dict['beta_gate'] = z
    return 1 - z, z


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate update gates
    u, z = self._get_coupled_gates(x, prev_s)

    # - Calculate s_bar
    if self._use_reset_gate:
      s_bar, r = self.reset_14(
        x, prev_s, 's_bar', self._activation, reset_s=True, return_gate=True)
      self._gate_dict['reset_gate'] = r
    else: s_bar = self.dense_rn(x, prev_s, 's_bar', self._activation)

    # - Update state
    with tf.name_scope('transit'):
      new_s = z * prev_s + u * s_bar

    # - Calculate output and return
    y = new_s
    return y, new_s



