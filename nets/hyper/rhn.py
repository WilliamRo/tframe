from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub as th
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.hyper_kernel import HyperKernel


class RHN(CellBase, HyperKernel):
  """Generalized Recurrent Highway Network
     Reference:
       [1] Zilly, etc. Recurrent Highway Networks
  """
  net_name = 'rhn'

  def __init__(
      self,
      state_size,
      num_layers,
      hyper_kernel,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      forget_bias=0,
      **kwargs):
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer, use_bias,
                      bias_initializer, **kwargs)

    self.kernel_key = checker.check_type(hyper_kernel, str)
    # Specific attributes
    self._state_size = checker.check_positive_integer(state_size)
    self._num_layers = checker.check_positive_integer(num_layers)
    self._hyper_kernel = self._get_hyper_kernel(
      hyper_kernel, do=th.rec_dropout, ln=self._layer_normalization,
      forget_bias=forget_bias)


  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = self._get_hyper_state_holder(
      self.kernel_key, self._state_size)
    return self._init_state


  @property
  def _scale_tail(self):
    return '_{}({}x{})'.format(
      self.kernel_key, self._state_size, self._num_layers)


  @staticmethod
  def mark():
    return 'rhn_{}({}x{})'.format(th.hyper_kernel, th.state_size, th.num_layers)


  def _link(self, s, x, **kwargs):
    for i in range(self._num_layers):
      with tf.variable_scope('layer_{}'.format(i + 1)):
        output, s = self._hyper_kernel(x if i == 0 else None, s)

    return output, s



