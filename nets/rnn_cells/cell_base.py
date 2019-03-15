from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import activations
from tframe import checker
from tframe import initializers
from tframe.nets import RNet


class CellBase(RNet):
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
      **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)

    # Common attributes
    self._activation = activations.get(activation, **kwargs)
    self._weight_initializer = initializers.get(weight_initializer)
    self._use_bias = checker.check_type(use_bias, bool)
    self._bias_initializer = initializers.get(bias_initializer)

    self._output_scale_ = None

  @property
  def _output_scale(self):
    if self._output_scale_ is not None: return self._output_scale_
    return self._state_size

  # TODO: this property is a compromise to avoid error in Net.
  @_output_scale.setter
  def _output_scale(self, val): self._output_scale_ = val

  @property
  def _scale_tail(self): raise NotImplemented

  def structure_string(self, detail=True, scale=True):
    return self.net_name + self._scale_tail if scale else ''

