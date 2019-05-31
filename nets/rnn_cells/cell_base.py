from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.nets import RNet
from tframe.operators.apis.neurobase import RNeuroBase


class CellBase(RNet, RNeuroBase):
  """Base class for RNN cells.
     TODO: all rnn cells are encouraged to in inherit this class
  """
  net_name = 'cell_base'

  def __init__(
      self,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      layer_normalization=False,
      **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)
    RNeuroBase.__init__(self, activation, weight_initializer, use_bias,
                        bias_initializer, layer_normalization, **kwargs)

    self._output_scale_ = None

  # region : Properties

  @property
  def _output_scale(self):
    if self._output_scale_ is not None: return self._output_scale_
    return self._state_size

  # TODO: this property is a compromise to avoid error in Net.
  @_output_scale.setter
  def _output_scale(self, val): self._output_scale_ = val

  @property
  def _scale_tail(self):
    assert self._state_size is not None
    return '({})'.format(self._state_size)

  def structure_string(self, detail=True, scale=True):
    return self.net_name + self._scale_tail if scale else ''

  # endregion : Properties
