"""This module is based on this paper:
Quan, et al., Self2Self With Dropout: Learning Self-Supervised Denoising From
Single Image, CVF, 2020.
"""
from tframe import tf
from tframe import pedia
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input



class InputDropout(Layer):
  """Input dropout"""
  abbreviation = 'in_drop'
  full_name = 'input_dropout'

  def __init__(self, drop_prob=0.5, force_mask=False):
    self.drop_prob = drop_prob
    self.drop_mask = None

    self.force_mask = force_mask

  @property
  def structure_tail(self):
    return '({:.2f})'.format(1 - self.drop_prob)

  @property
  def is_training(self): return tf.get_collection(pedia.is_training)[0]

  # region: Link

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    """This implementation is based on tf.nn.dropout_v2"""
    # Note that x.shape may be partially unknown
    random_tensor = tf.random.uniform(tf.shape(x))

    keep_mask = random_tensor >= self.drop_prob
    keep_mask = tf.cast(keep_mask, x.dtype)
    masked_x = x * keep_mask

    # Save drop_mask for loss kernel
    self.drop_mask = 1.0 - keep_mask

    if self.force_mask: return masked_x
    return tf.cond(self.is_training, lambda: masked_x, lambda: x)

  # endregion: Link

  # region: Loss

  def get_loss(self, loss='mse'):
    from tframe.core.quantity import Quantity
    assert loss in ('mse', 'mae')

    func = tf.square if loss == 'mse' else tf.abs

    def kernel(y_true, y_pred):
      assert isinstance(self.drop_mask, tf.Tensor)
      delta = func(y_true - y_pred)
      masked_delta = self.drop_mask * delta
      if self.force_mask: return masked_delta
      return tf.cond(self.is_training, lambda: masked_delta, lambda: delta)

    return Quantity(kernel, tf_summ_method=tf.reduce_mean,
                    name=f'M-{loss.upper()}')

  # endregion: Loss

