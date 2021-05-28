from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.operators.apis.generic_neurons import GenericNeurons as gn

import tframe.nets.rnn_cells.cell_base as cell_base


class DynamicWeights(object):

  def neurons_with_reset_gate(self, x, s, reset_who, return_with_gate=False):
    # Sanity check
    assert reset_who in ('a', 's') and isinstance(self, cell_base.CellBase)
    # Calculate reset gate
    r = self.neurons(x, s, is_gate=True, scope='reset_gate')
    self._gate_dict['reset_gate'] = r
    # Calculate s_bar
    if reset_who == 'a':
      with tf.variable_scope('s_bar'):
        a_s = gn.psi(self._state_size, s, use_bias=False, suffix='s',
                     weight_initializer=self._weight_initializer)
        # Reset a
        a_s = r * a_s
        s_bar = gn.psi(self._state_size, x, use_bias=True,
                       bias_initializer=self._bias_initializer,
                       weight_initializer=self._weight_initializer, suffix='x',
                       additional_summed_input=a_s)
      if self._activation: s_bar = self._activation(s_bar)
    else:
      # Reset s
      s = r * s
      s_bar = self.neurons(x, s, activation=self._activation, scope='s_bar')

    # Return
    if return_with_gate: return s_bar, r
    else: return s_bar

