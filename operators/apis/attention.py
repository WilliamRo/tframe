from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import hub as th
from tframe import checker
from tframe.operators.apis.neurobase import NeuroBase


class AttentionBase(NeuroBase):
  """
  Implementation Reference:
  [1] Ashish Vaswani, etc. "Attention Is All You Need". 2017.
  [2] https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb#scrollTo=LazzUq3bJ5SH
  """

  @staticmethod
  def scaled_dot_prod_attention(Q, K, V, mask=None, return_a=False):
    r"""Calculate the attention weights.

    Q.shape = (..., seq_len_q, depth)
    K.shape = (..., seq_len_k, depth)
    V.shape = (..., seq_len_v, depth_v)
    If mask is not None, mask.shape should be broadcastable to
    (..., seq_len_q, seq_len_k)

    If Q and K have a mean of 0 and variance of 1, their matrix
    multiplication will have a mean of 0 and variance of dk.
    (see if \xi and \eta follow N(0, 1), dk*\xi*\eta follows N(0, dk))
    Thus the dot-product attention is scaled by a factor of square root of dk.
    """
    # Output shape = (..., seq_len_q, seq_len_k)
    matmul_QK = tf.matmul(Q, K, transpose_b=True)

    # Scale matmul_QK
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_QK / tf.math.sqrt(dk)

    # Add mask
    if mask is not None: scaled_attention_logits += (mask * -1e9)

    # Calculate the attention weights and output
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)

    # Return accordingly
    # output.shape = (.., seq_len_q, depth_v)
    # attention_weights.shape = (..., seq_len_q, seq_len_k)
    if return_a: return output, attention_weights
    else: return output


  @staticmethod
  def _check_qkv(q, k, v):
    # Check type
    assert all([isinstance(t, tf.Tensor) for t in (q, k, v)])
    # Check shape
    q_shape, k_shape, v_shape = [t.shape.as_list() for t in (q, k, v)]
    assert all([len(shape) > 2 for shape in (q_shape, k_shape, v_shape)])
    # Make sure keys and values have the same length
    q_len, k_len, v_len = [s[-2] for s in (q_shape, k_shape, v_shape)]
    assert k_len == v_len
    # Return recommended QK_dim and lengths
    return min(q_shape[-1], k_shape[-1]), q_len, k_len, v_len


  @staticmethod
  def _split_heads(x, length, num_heads, depth):
    """Split the last dimension into (num_heads, depth)
       Input shape:  (bs, length, num_heads * depth)
       Output shape: (bs, num_heads, length, depth)"""
    x = tf.reshape(x, (-1, length, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])


  def _mha(self, q, k, v, num_heads=1, QK_dim=None, V_dim=None,
           output_dim=None, mask=None):
    """Multi-head attention.
       Must be called within a separate scope when multiple _mhas are called
       inside the same master scope."""
    # Check q, k, v and set default QK_dim if not provided
    min_qk_dim, q_len, k_len, v_len = self._check_qkv(q, k, v)
    if QK_dim is None: QK_dim = min_qk_dim

    # Calculate Q, K, V, where Q, K must be transformed by q and k
    Q = self.dense(num_heads * QK_dim, q, scope='query')
    K = self.dense(num_heads * QK_dim, k, scope='key')
    # V is allowed to be v
    if V_dim is not None: V = self.dense(num_heads * V_dim, v, scope='value')
    elif num_heads == 1: V = v
    else: V = tf.stack([v] * num_heads, axis=-3)
    # TODO: if V_dim is not specified, simply stack v may not be appropriate.
    #       since the output attention may be grown large

    # Split head if necessary
    if num_heads > 1:
      Q = self._split_heads(Q, q_len, num_heads, QK_dim)
      K = self._split_heads(K, k_len, num_heads, QK_dim)
      if V_dim is not None: V = self._split_heads(V, v_len, num_heads, V_dim)

    # Apply attention, out shape = (bs[, num_heads], q_len, [Vv]_dim)
    attention = self.scaled_dot_prod_attention(Q, K, V, mask)

    # Reshape back if necessary
    if num_heads > 1:
      attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # type: tf.Tensor
      last_dim = num_heads * attention.shape.as_list()[-1]
      attention = tf.reshape(attention, (-1, q_len, last_dim))

    # Calculate output and return
    output = (attention if output_dim is None
              else self.dense(output_dim, attention, scope='output'))
    return output

