from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe.nets.rnn_cells.cell_base import CellBase
from tframe.operators.apis.distributor import Distributor


class Mask1D(CellBase):

  def column_mask(self, scope, *inputs, seeds=None, activation=None,
                  gate_mask=True, num_neurons=None):
    # Check num_neurons
    if num_neurons is None: num_neurons = self._state_size
    else: assert isinstance(num_neurons, int) and num_neurons > 0
    # Check seed
    if seeds is None: seeds = inputs
    if not isinstance(seeds, (tuple, list)): seeds = [seeds]

    masked_inputs = []
    with tf.variable_scope(scope):
      for i, x in enumerate(inputs):
        assert isinstance(x, tf.Tensor)
        mask = self._get_mask(x, seeds, gate_mask, 'mask{}'.format(i + 1))
        masked_inputs.append(x * mask)

    na = self.differentiate(num_neurons, scope, activation=activation)
    return na(*masked_inputs)


  def _get_mask(self, x, seeds, gate_mask, name):
    assert isinstance(x, tf.Tensor)
    na = self.differentiate(self.get_dimension(x), name,
                            use_bias=gate_mask, is_gate=gate_mask)
    return na(*seeds)



