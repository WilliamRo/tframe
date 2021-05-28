from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import hub
from tframe import pedia
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
      dropout_rate=0.0,
      zoneout_rate=0.0,
      **kwargs):

    # Call parent's constructor
    RNet.__init__(self, self.net_name)
    RNeuroBase.__init__(
      self,
      activation=activation,
      weight_initializer=weight_initializer,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      layer_normalization=layer_normalization,
      zoneout_rate=zoneout_rate,
      dropout_rate=dropout_rate,
      **kwargs)

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

  def _get_s_bar(self, x, s, output_dim=None, use_reset_gate=False):
    if output_dim is None: output_dim = self._state_size
    if use_reset_gate:
      r = self.dense_rn(
        x, s, 'reset_gate', output_dim=self.get_dimension(s), is_gate=True)
      self._gate_dict['reset_gate'] = r
      s = r * s
    return self.dense_rn(x, s, 's_bar', self._activation, output_dim=output_dim)

  def _gast(self, pre_s, s_bar, update_gate=None, forget_gate=None):
    """Gated Additive State Transition."""
    assert not all([update_gate is None, forget_gate is None])
    # Couple gates if necessary
    if forget_gate is None: forget_gate = tf.subtract(1.0, update_gate)
    elif update_gate is None: update_gate = tf.subtract(1.0, forget_gate)
    # Apply recurrent dropout without memory loss if necessary
    if self._dropout_rate > 0: s_bar = self.dropout(s_bar, self._dropout_rate)
    # Update states
    with tf.name_scope('GAST'): return tf.add(
        tf.multiply(forget_gate, pre_s), tf.multiply(update_gate, s_bar))

  @classmethod
  def _zoneout(cls, new_s, prev_s, ratio):
    def zo(n_s, p_s, r):
      if r == 0: return n_s
      assert cls.get_dimension(n_s) == cls.get_dimension(p_s) and 0 < r < 1
      seed = tf.random_uniform(tf.shape(n_s), 0, 1)
      z = tf.cast(tf.less(seed, r), hub.dtype)
      result = z * p_s + (1. - z) * n_s
      return tf.cond(tf.get_collection(
        pedia.is_training)[0], lambda: result, lambda: n_s)
    if not isinstance(new_s, (tuple, list)): new_s = [new_s]
    if not isinstance(prev_s, (tuple, list)): prev_s = [prev_s]
    if not isinstance(ratio, (tuple, list)): ratio = [ratio]
    outputs = [zo(n_s, p_s, r) for n_s, p_s, r in zip(new_s, prev_s, ratio)]
    if len(outputs) == 1: return outputs[0]
    else: return tuple(outputs)


