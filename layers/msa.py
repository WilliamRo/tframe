from tframe import tf

from tframe.layers.layer import single_input, Layer
from tframe.operators.apis.attention import AttentionBase



class MultiHeadSelfAttention(AttentionBase, Layer):

  full_name = 'attention'
  abbreviation = 'msa'

  def __init__(self, num_heads, use_keras=False, dropout=0.0):
    super().__init__()
    self.num_heads = num_heads
    self.use_keras = use_keras
    self.dropout = dropout


  @property
  def structure_tail(self):
    return f'(h={self.num_heads})'


  @single_input
  def _link(self, x: tf.Tensor):
    # TODO: dropout is to be considered

    # Input shape = [B, seq_len, D]
    x_shape = x.shape.as_list()
    D = x_shape[-1]

    if self.use_keras:
      from tensorflow.keras.layers import MultiHeadAttention
      from tframe import pedia

      training = tf.get_collection(pedia.is_training)[0]
      layer = MultiHeadAttention(self.num_heads, key_dim=D,
                                 dropout=self.dropout)
      x = layer(x, x, training=training)
    else:
      x = self._mha(x, x, x, num_heads=self.num_heads)

    return x
