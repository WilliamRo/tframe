from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub as th
from tframe.nets.rnn_cells.cell_base import CellBase
from tframe.operators.apis.gam import GAM
from tframe.operators.apis.hyper_kernel import HyperKernel


class GamRHN(CellBase, GAM, HyperKernel):

  net_name = 'gam_rhn'

  def __init__(
      self,
      gam_config,
      head_size,
      state_size,
      num_layers=1,
      kernel='gru',
      activation='tanh',
      gam_dropout=0.0,
      rhn_dropout=0.0,
      use_reset_gate=True,
      **kwargs):
    # Call parent's constructor
    CellBase.__init__(self, activation=activation, **kwargs)
    GAM.__init__(self, gam_config, head_size)
    # Own attributes
    self._gam_dropout = checker.check_gate(gam_dropout)
    self._rhn_dropout = checker.check_gate(rhn_dropout)
    self._state_size = checker.check_positive_integer(state_size)
    self._num_layers = checker.check_positive_integer(num_layers)
    self._kernel_key = checker.check_type(kernel, str)
    self._highway_op = self._get_hyper_kernel(kernel, do=self._rhn_dropout)
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)


  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = (
      self._get_placeholder('gam', self.total_size),
      self._get_hyper_state_holder(self._kernel_key, self._state_size))
    return self._init_state


  @property
  def _scale_tail(self):
    option = ''
    if self._use_reset_gate: option = 'r'
    if option: option = '[{}]'.format(option)
    return '({}-H{}|{}{}xL{}){}'.format(
      self.group_string, self._head_size, self._kernel_key,
      self._state_size, self._num_layers, option)


  @staticmethod
  def mark():
    option = ''
    if th.use_reset_gate: option = 'r'
    if option: option = '({})'.format(option)
    return 'gam({}H{})rhn({}{}xL{}){}'.format(
      th.gam_config, th.head_size,
      th.hyper_kernel, th.state_size, th.num_layers, option)


  def _link(self, prev_states, x, **kwargs):
    gam, state = prev_states
    output = state[0] if isinstance(state, (tuple, list)) else state
    self._reset_counter() # important

    # - Write to GAM
    gam, hw = self._write(gam, x, output, dropout=self._gam_dropout)

    # - Highway
    for i in range(self._num_layers):
      x_tilde = self._read(gam, output, head=hw if i == 0 else None)
      with tf.variable_scope('highway_{}'.format(i + 1)):
        output, state = self._highway_op(x_tilde, state)

    return output, (gam, state)


