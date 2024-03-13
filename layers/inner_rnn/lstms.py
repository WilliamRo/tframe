from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import context
from tframe import checker
from tframe import hub as th
from tframe.layers.hyper.hyper_base import HyperBase


class BiLSTM(HyperBase):

  full_name = 'bilstm'
  abbreviation = 'bilstm'

  def __init__(
      self,
      state_size,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      **kwargs):
    """
    Softmax over groups applied to neurons.
    Case 1: head_size < 0: does not use extra neurons
    Case 2: head_size = 0: use extra neurons without a head
    Case 3: head_size > 0: use extra neurons with a head
    """
    # Call parent's constructor
    super().__init__(None, weight_initializer, use_bias,
                     bias_initializer, **kwargs)

    # Specific attributes
    self._state_size = state_size


  @property
  def structure_tail(self):
    return f'({self._state_size})'


  def forward(self, x, **kwargs):
    # x.shape = [bs, L, C]

    # x.shape = [bs, L, C]
    y = None
    return y
