from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub as th
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.hyper_kernel import HyperKernel


class FastSlow(CellBase, HyperKernel):
  """Fast-slow RNN
     Reference:
       [1] Fast-Slow Recurrent Neural Networks, 2017
   """
  net_name = 'fs'

  def __init__(
      self,
      fast_size,
      fast_layers,
      slow_size,
      hyper_kernel,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      input_dropout=0.0,
      output_dropout=0.0,
      forget_bias=0,
      **kwargs):
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer, use_bias,
                      bias_initializer, **kwargs)

    self.kernel_key = checker.check_type(hyper_kernel, str)
    # Specific attributes
    self._fast_size = checker.check_positive_integer(fast_size)
    self._fast_layers = checker.check_positive_integer(fast_layers)
    self._slow_size = checker.check_positive_integer(slow_size)
    self._hyper_kernel = self._get_hyper_kernel(
      hyper_kernel, do=th.rec_dropout, forget_bias=forget_bias)

    self._input_do = checker.check_type(input_dropout, float)
    self._output_do = checker.check_type(output_dropout, float)


  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = (
      self._get_hyper_state_holder(self.kernel_key, self._fast_size),
      self._get_hyper_state_holder(self.kernel_key, self._slow_size))
    return self._init_state


  @property
  def _scale_tail(self):
    return '{}({}x{}+{})'.format(
      self.kernel_key, self._fast_size, self._fast_layers, self._slow_size)


  @staticmethod
  def mark():
    return 'fs{}({}x{}+{})'.format(
      th.hyper_kernel, th.fast_size, th.fast_layers, th.slow_size)


  def _link(self, prev_s, x, **kwargs):
    f_state, s_state = prev_s
    prev_f_state, prev_s_state = prev_s
    if self._input_do > 0: x = self.dropout(x, self._input_do)

    with tf.variable_scope('fast_0'):
      f_output, f_state = self._hyper_kernel(x, f_state)

    with tf.variable_scope('slow'):
      s_output, s_state = self._hyper_kernel(f_output, s_state)

    with tf.variable_scope('fast_1'):
      f_output, f_state = self._hyper_kernel(s_output, f_state)

    for i in range(2, self._fast_layers):
      with tf.variable_scope('fast_{}'.format(i)):
        f_output, f_state = self._hyper_kernel(None, f_state)

    if self._output_do > 0: f_output = self.dropout(f_output, self._output_do)

    if self.kernel_key in ['lstm', 'cplstm']:
      assert len(f_state) == len(s_state) == 2
      ratio = (th.hidden_zoneout, th.cell_zoneout)
      f_state = self._zoneout(f_state, prev_f_state, ratio)
      s_state = self._zoneout(s_state, prev_s_state, ratio)
    return f_output, (f_state, s_state)



