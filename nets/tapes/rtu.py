from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe.operators.apis.attention import AttentionBase
from tframe.nets.tapes.tape import Tape


class RTU(Tape, AttentionBase):
  """Recurrent Tape Unit"""

  net_name = 'rtu'

  def __init__(
      self,
      state_size,
      tape_length,
      spatial_size=None,
      spatial_heads=1,
      temporal_heads=1,
      spatial_dropout=0.0,
      temporal_dropout=0.0,
      activation='tanh',
      weight_initializer='xavier_uniform',
      use_bias=True,
      bias_initializer='zeros',
      rec_dropout=0.0,
      **kwargs):
    """
    """
    # Call parent's constructor
    super().__init__(
      length=tape_length, depth=state_size, num_tapes=1,
      activation=activation, weight_initializer=weight_initializer,
      use_bias=use_bias, bias_initializer=bias_initializer,
      dropout_rate=rec_dropout, **kwargs)
    # Specific attributes
    self._state_size = state_size
    self._spatial_size = spatial_size
    self._spatial_heads = checker.check_positive_integer(spatial_heads)
    self._temporal_heads = checker.check_positive_integer(temporal_heads)
    self._spatial_dropout = spatial_dropout
    self._temporal_dropout = temporal_dropout

    # assert self._state_size % self._temporal_heads == 0
    # self._temporal_dim = self._state_size // self._temporal_heads

  # region : Process

  def _process(self, tape, inp):
    """While the Tape allows inp of single time-step, it is forbidden here."""
    # Sanity check
    assert isinstance(tape, tf.Tensor) and isinstance(inp, tf.Tensor)
    assert all([len(x.shape) > 2 for x in (tape, inp)])

    # Prepare query seed
    prev_s, curr_x = tape[..., -1:, :], inp[..., -1:, :]
    query = tf.concat([prev_s, curr_x], axis=-1)

    # Create mask
    padding_mask = None

    # Apply spatial attention
    with tf.variable_scope('spatial'):
      x = self.spatial_attention(query, inp, padding_mask, curr_x)

    # Apply temporal attention
    with tf.variable_scope('temporal'):
      # TODO: decide to use x or query as q for temporal_attention
      s = self.temporal_attention(query, tape, padding_mask)

    # Apply state transition
    new_state = self.transit(s, x, method='gast')

    # Return new state
    return new_state

  def spatial_attention(self, q, inp, mask, curr_x):
    # Determine spatial dim
    if self._spatial_size is None: self._spatial_size = inp.shape.as_list()[-1]
    assert self._spatial_size % self._spatial_heads == 0
    # spatial_dim = self._spatial_size // self._spatial_heads
    # Apply attention mechanism
    attn = self._mha(
      q, inp, inp, num_heads=self._spatial_heads, QK_dim=None, V_dim=None,
      output_dim=self._spatial_size, mask=mask)
    # Dropout if necessary
    if self._spatial_dropout > 0:
      attn = self.dropout(attn, self._spatial_dropout)
    # Apply layer normalization, out shape = (bs, 1, spatial_size)
    # TODO: this requires 'spatial_size == curr_x.shape[-1]'
    x = self.layer_normalize(attn + curr_x, epsilon=1e-6) # TODO <=
    return x

  def temporal_attention(self, q, tape, mask):
    """Generate s"""
    # Do not transform the weighted sum of values in temporal mha TODO
    # (1) determine V_dim
    V_dim = None
    # assert self._state_size % self._temporal_heads == 0
    # V_dim = self._state_size // self._temporal_heads
    # (2) determine output
    output_dim = None
    # output_dim = tape.shape.as_list()[-1]
    # (*) Apply multi-head attention
    attn = self._mha(
      q, tape, tape, num_heads=self._temporal_heads, QK_dim=None, V_dim=V_dim,
      output_dim=output_dim, mask=mask)
    # Dropout if necessary
    if self._temporal_dropout > 0:
      attn = self.dropout(attn, self._temporal_dropout)
    # (3) Post process TODO
    attn = self.layer_normalize(attn + tape[..., -1:, :], epsilon=1e-6)
    return attn

  def transit(self, s, x, method='srn'):
    # Preparation
    squeeze = lambda t: tf.squeeze(t, axis=-2)
    s_bar = None
    if method in ['srn', 'gast']:
      s, x = [squeeze(t) for t in (s, x)]
      s_bar = self.dense_v2(self._state_size, 'srn_transit', s, x)
      # TODO
      # s_bar = self.layer_normalize(s_bar)

    # Transit accordingly
    if method in ['srn']: return s_bar
    elif method in ['gast']:
      g = self.dense_v2(self._state_size, 'update_gate', s, x, is_gate=True)
      return self._gast(s, s_bar, update_gate=g)
    else: raise KeyError('Unknown method `{}`'.format(method))

  # endregion : Process


"""
From the perspective of the integrated model, denote the input shape as 
[bs, in_dim].
Data fed into RT layer should be of shape [bs, L, in_dim], which is guaranteed
by the tape layer (conveyor).
At time $t$, the state transition can be written as
... ...





"""
