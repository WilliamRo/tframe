from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import checker
from tframe import hub as th

from tframe.operators.apis.neurobase import RNeuroBase
from tframe.operators.apis.attention import AttentionBase


class GAM(RNeuroBase, AttentionBase):
  """Grouped Auxiliary Memory (Version 2.0)"""

  def __init__(self, num_channels, channel_size, read_head_size,
               write_head_size=None):
    self._num_channels = checker.check_positive_integer(num_channels)
    self._channel_size = checker.check_positive_integer(channel_size)
    self._read_head_size = checker.check_positive_integer(read_head_size)
    if write_head_size is None: write_head_size = read_head_size
    self._write_head_size = checker.check_positive_integer(write_head_size)

    self._gam_tensor = None

  # region : Properties

  @property
  def gam_shape(self): return [self._num_channels, self._channel_size]

  @property
  def _gam(self): return self._check_gam(self._gam_tensor)

  # endregion : Properties

  # region : Addressing

  def _address_by_location(self, head):
    # Check head
    assert isinstance(head, tf.Tensor)
    head_shape = head.shape.as_list()
    assert len(head_shape) == 2
    dim_head = head_shape[1]

    return None

  # endregion : Addressing

  # region : Read Operations

  def _read_by_location(
      self, *inputs, head=None, return_head=False, identifier=''):
    """
    inputs: list of rank-2 tf.Tensor each of which has shape [None, whatever]
    """
    # Sanity check
    assert not all([len(inputs) == 0, head is None])
    if len(inputs) > 0: checker.check_type(inputs, tf.Tensor)
    # Generate read head if necessary
    if head is None:
      head_scope = 'read_head'
      if identifier: head_scope += '_{}'.format(identifier)
      head = self.dense_v2(self._read_head_size, head_scope, *inputs)

    # Calculate address
    a = self._address_by_location(head)

    y = None
    # Return accordingly
    if return_head: return y, head
    else: return y

  def _read_by_content(self):
    raise NotImplementedError

  # endregion : Read Operations

  # region : Write Operations

  def _write_by_location(self, head=None, return_head=False):
    pass

  def _write_by_content(self):
    raise NotImplementedError

  # endregion : Write Operations

  # region : Other Private Methods

  def _check_gam(self, gam_tensor):
    assert isinstance(gam_tensor, tf.Tensor)
    assert tuple(gam_tensor.shape.as_list()[1:]) == tuple(self.gam_shape)
    return gam_tensor

  def _link_to_device(self, gam_tensor):
    # assert self._gam_tensor is None
    self._gam_tensor = self._check_gam(gam_tensor)

  def _get_init_gam(self):
    """This method should be called inside successor's init_state property.
       TODO: should be unified with _get_placeholder in rnet.py
    """
    return tf.placeholder(
      dtype=th.dtype, shape=[None] + self.gam_shape, name='gam')

  # endregion : Other Private Methods

