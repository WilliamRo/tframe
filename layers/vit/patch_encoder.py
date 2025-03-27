from tframe import tf
from tframe.layers.layer import Layer, single_input
from tframe.operators.neurons import NeuroBase

import numpy as np



class PatchEncoder(Layer, NeuroBase):
  """Flatten image patches and add positional embedding"""

  full_name = 'patch_encoder'
  abbreviation = 'pe'


  def __init__(self, dim, use_positional_embedding=True):
    super().__init__()

    self.dim = dim
    self.use_positional_embedding = use_positional_embedding
    self.embedding = None


  @property
  def structure_tail(self):
    s = f'{self.dim}'
    if not self.use_positional_embedding: s += ',NPE'
    return f'({s})'


  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    from tensorflow.keras.layers import Embedding

    # ATTENTION 1: currently only 2D images are supported

    x_shape = x.shape.as_list()
    # x_shape = [B, n_patches, h, w, C]
    assert len(x_shape) == 5

    # (1) Project patches to embedding space -> [B, n_patches, dim]
    n_patches, h, w, C = x_shape[1:]
    x_flattened = tf.reshape(x, [-1, n_patches, h * w * C])
    x_projected = self.dense(self.dim, x_flattened, scope='projector')

    # (2) Add positional embedding
    if self.use_positional_embedding:
      # Create positional embedding
      # positions.shape = [1, n_patches]
      positions = tf.expand_dims(tf.range(0, n_patches), axis=0)

      # pe.shape = [1, n_patches, dim]
      self.embedding = Embedding(n_patches, self.dim)
      pe = self.embedding(positions)
      x_projected += pe

    # Output shape = [B, n_patches, dim]
    return x_projected
