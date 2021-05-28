from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

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
    (1) output_size is 0 or None
        y = new_s
    (2) output_size is not None
        y = neuron(x, prev_s, ...)
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
    self._output_size = None if output_size == 0 else output_size
    self._spatial_groups = []
    if spatial_configs is not None:
      output_dim = (self._output_size if self._output_size is not None
                    else self._state_size)
      # Set spatial_groups
      if spatial_configs == 'default':
        # Check output size
        num_groups = output_dim // 2
        assert num_groups * 2 == output_dim
        self._spatial_groups = [(2, num_groups, 1)]
      else:
        assert isinstance(spatial_configs, str) and len(spatial_configs) > 0
        self._spatial_groups = self._get_groups(spatial_configs)
        total_size = self._get_total_size(self._spatial_groups)
        # Check output dim
        assert output_dim == total_size

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
    y = new_s
    if self._output_size is not None:
      assert self._output_size > 0
      y = self.neurons(
        x, prev_s, num=self._output_size,
        activation=self._spatial_activation, scope='y_bar')

    if len(self._spatial_groups) > 0:
      # Check input size
      input_size = linker.get_dimension(x)
      output_size = linker.get_dimension(y)
      assert input_size == output_size
      s_alpha, s_beta = self._get_coupled_gates(
        x, prev_s, self._spatial_groups, self._reverse_s)
      with tf.name_scope('output'):
        y = tf.add(tf.multiply(s_beta, x), tf.multiply(s_alpha, y))

    # - Return output and new state
    return y, new_s


