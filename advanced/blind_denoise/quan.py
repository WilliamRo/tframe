"""This module is based on this paper:
Quan, et al., Self2Self With Dropout: Learning Self-Supervised Denoising From
Single Image, CVF, 2020.
"""
from tframe import tf
from tframe import pedia
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.core.nomear import Nomear

import numpy as np



class InputDropout(Layer, Nomear):
  """Input dropout"""
  abbreviation = 'in_drop'
  full_name = 'input_dropout'

  def __init__(self, drop_prob=0.5, mask_size=1, force_mask=False,
               mask_gen_method='tf'):
    self.drop_prob = drop_prob
    self.drop_mask = None
    self.mask_size = mask_size

    self.force_mask = force_mask

    assert mask_gen_method in ('tf', 'np')
    self.mask_gen_method = mask_gen_method

  # region: Properties

  @property
  def structure_tail(self):
    return '({:.2f}sz{})'.format(1 - self.drop_prob, self.mask_size)

  @property
  def is_training(self): return tf.get_collection(pedia.is_training)[0]

  # endregion: Properties

  # region: Mask Generation

  def _tf_gen_mask(self, x: tf.Tensor):
    if self.mask_size == 1:
      random_tensor = tf.random.uniform(tf.shape(x))
    else:
      x_shape = tf.shape(x)
      dim = x_shape.shape[0] - 2
      shape_tensor = tf.stack(
        [x_shape[0]] + [x_shape[i+1] // self.mask_size for i in range(dim)]
        + [x.shape.as_list()[-1]])
      random_tensor = tf.random.uniform(shape_tensor)

      if dim == 2:
        random_tensor = tf.image.resize_images(
          random_tensor, size=[x_shape[i+1] for i in range(dim)],
          method='nearest')
      elif dim == 3:
        # For 3-D volumes, depth, width, and height must be multiples of
        # `mask_size`
        random_tensor = tf.keras.backend.resize_volumes(
          random_tensor, self.mask_size, self.mask_size, self.mask_size,
          data_format='channels_last')
      else: raise NotImplementedError

      # Randomly translate mask
      for i in range(dim):
        shift = tf.random.uniform((), 0, self.mask_size, dtype=tf.int32)
        random_tensor = tf.roll(random_tensor, shift, i+1)

    keep_mask = random_tensor >= self.drop_prob
    keep_mask = tf.cast(keep_mask, x.dtype)
    return keep_mask

  def _get_mask_placeholder(self, x: tf.Tensor):
    def init():
      from tframe import hub as th
      shape = x.shape.as_list()
      keep_mask = tf.placeholder(dtype=th.dtype, shape=shape, name='keep_mask')
      return keep_mask
    return self.get_from_pocket('keep_mask', initializer=init)

  def _np_gen_mask(self, x: np.ndarray):
    """x.shape = [batch, (depth, )height, width, channels]"""
    if self.mask_size == 1:
      random_array = np.random.uniform(size=x.shape)
    else:
      dim = len(x.shape) - 2
      random_shape = [x.shape[0]] + [
        x.shape[i+1] // self.mask_size for i in range(dim)] + [x.shape[-1]]
      random_array = np.random.uniform(size=random_shape)

      # Resize back to x.shape
      for i in range(dim):
        random_array = random_array.repeat(self.mask_size, axis=i+1)

      # Randomly translate mask
      for i in range(dim):
        random_array = np.roll(
          random_array, np.random.randint(0, self.mask_size), axis=i+1)

    return (random_array >= self.drop_prob).astype(np.float32)

  # endregion: Mask Generation

  # region: Link

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    """This implementation is based on tf.nn.dropout_v2"""
    # Note that x.shape (N, (D, )H, W, C) may be partially unknown
    # Generate mask accordingly
    if self.mask_gen_method == 'tf': keep_mask = self._tf_gen_mask(x)
    else: keep_mask = self._get_mask_placeholder(x)

    # Mask input
    masked_x = x * keep_mask

    # Save drop_mask for loss kernel
    self.drop_mask = 1.0 - keep_mask

    from tframe import context
    context.add_to_dict_collection('quan', 'masked_input', masked_x)
    context.add_to_dict_collection('quan', 'drop_mask', self.drop_mask)

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



if __name__ == '__main__':
  id = InputDropout(mask_size=3, mask_gen_method='np')
  x = np.zeros(shape=[10, 9, 9, 9, 1])
  print(id._np_gen_mask(x)[0, ..., 0])
