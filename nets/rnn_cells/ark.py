from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import checker
from tframe import linker
from tframe.nets.rnn_cells.cell_base import CellBase
from tframe.nets.rnn_cells.gdu import GDU


class Noah(GDU):
  """Noah"""
  net_name = 'noah'

  def __init__(
      self,
      temporal_configs,
      output_size=None,
      spatial_configs=None,
      temporal_reverse=False,
      spatial_reverse=False,
      temporal_activation='tanh',
      spatial_activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      **kwargs):
    """
    :param output_size:
    Denote y as cell output, s as state
    (1) output_size is None
       (1.1) spatial_configs is None
             y = new_s
       (1.2) spatial_configs is 'default'
             output_size = state_size
       (1.3) spatial_configs is a list or non-empty string
             output_size will be determined by the total_size specified by
             spatial_configs
    (2) output_size is not None
       (2.1) spatial_configs is None
             y = neuron(x, prev_s, ...)
       (2.2) spatial_configs is 'default'
             ASSERTION - output_size should be an positive even integer
       (2.3) spatial_configs is a list or non-empty string
             ASSERTION - output_size should be equal to the total_size specified
             by spatial_configs
    """
    # Call parent's constructor
    CellBase.__init__(self, temporal_activation, weight_initializer,
                      use_bias, bias_initializer, **kwargs)
    self._temporal_activation = self._activation
    self._spatial_activation = spatial_activation

    # Specific attributes
    self._temporal_groups = self._get_groups(temporal_configs)
    self._state_size = self._get_total_size(self._temporal_groups)
    # Set spatial groups
    self._spatial_groups = []
    if spatial_configs is not None:
      # Set spatial_groups
      if spatial_configs == 'default':
        # Check output size
        if output_size is None: output_size = self._state_size
        num_groups = output_size // 2
        assert num_groups * 2 == output_size
        self._spatial_groups = [(2, num_groups, 1)]
      else:
        assert isinstance(spatial_configs, str) and len(spatial_configs) > 0
        self._spatial_groups = self._get_groups(spatial_configs)
        total_size = self._get_total_size(self._spatial_groups)
        # Check output size
        if output_size is None: output_size = total_size
        else: assert output_size == total_size
    # Set output size
    self._output_size = output_size

    self._reverse_t = checker.check_type(temporal_reverse, bool)
    self._reverse_s = checker.check_type(spatial_reverse, bool)


  @property
  def _scale_tail(self):
    tail = ''
    # Spatial
    if self._output_size is not None:
      if len(self._spatial_groups) > 0:
        tail += '[{}]'.format(
          self._get_config_string(self._spatial_groups, self._reverse_s))
      else: tail += '[{}]'.format(self._output_size)
    # Temporal
    tail += '({})'.format(
      self._get_config_string(self._temporal_groups, self._reverse_t))
    return tail


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate new state
    # .. calculate alpha and beta gate
    t_alpha, t_beta = self._get_coupled_gates(
      x, prev_s, self._temporal_groups, self._reverse_t)
    # .. s_bar (candidate state)
    s_bar = self.neurons(
      x, prev_s, activation=self._temporal_activation, scope='s_bar')
    with tf.name_scope('state_transition'):
      new_s = tf.add(tf.multiply(t_beta, prev_s), tf.multiply(t_alpha, s_bar))

    # - Calculate output
    if self._output_size is None: y = new_s
    else:
      assert self._output_size > 0
      y = self.neurons(
        x, prev_s, num=self._output_size,
        activation=self._spatial_activation, scope='y_bar')
      if len(self._spatial_groups) > 0:
        # Check input size
        input_size = linker.get_dimension(x)
        assert input_size == self._output_size
        s_alpha, s_beta = self._get_coupled_gates(
          x, prev_s, self._spatial_groups, self._reverse_s)
        with tf.name_scope('output'):
          y = tf.add(tf.multiply(s_beta, x), tf.multiply(s_alpha, y))

    # - Return output and new state
    return y, new_s


