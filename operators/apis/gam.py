from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import checker
from tframe import hub

from tframe.operators.apis.groups import Groups
from tframe.operators.apis.neurobase import RNeuroBase


class GAM(Groups, RNeuroBase):
  """Grouped Auxiliary Memory"""

  def __init__(self, gam_config, head_size):
    # Call parent's constructor
    Groups.__init__(self, gam_config)
    # Additional attributes
    self._head_size = checker.check_positive_integer(head_size)
    # Prepare duplicating matrix and summarizing matrix
    self.D = None
    self.S = None
    # Do not init till link, otherwise D and S will be put into different graphs
    # self._init_const_matrices()
    # Be careful of inappropriate increment by while-free building
    # Use reset_counters at the beginning of _link
    self._head_counter = 0
    self._address_counter = 0

  # region : Properties

  @property
  def group_size(self):
    assert len(self._groups) == 1
    return self._groups[0][0]

  @property
  def head_scope(self):
    self._head_counter += 1
    return 'head_{}'.format(self._head_counter)

  @property
  def address_scope(self):
    self._address_counter += 1
    return 'address_{}'.format(self._address_counter)

  # endregion : Properties

  # region : Operations V2

  def _get_head(self, *inputs):
    head = self.dense_v2(
      self._head_size, self.head_scope, *inputs, use_bias=hub.head_bias)
    # Dropout may be applied
    return head

  def _get_address(self, *inputs, head=None, return_head=False):
    if head is None: head = self._get_head(*inputs)
    net_a = self.dense(
      self.total_size, head, self.address_scope, use_bias=hub.address_bias)
    a = self._softmax_over_groups(net_a)
    if return_head: return a, head
    return a

  def _transform(self, x, s):
    s_dim = self.get_dimension(s)
    r = self.dense_rn(x, s, 'reset_gate', is_gate=True, output_dim=s_dim)
    s = tf.multiply(r, s)
    return self.dense_rn(
      x, s, 'm_bar', activation='tanh', output_dim=self.num_groups)

  def _write(self, gam, x, s, dropout=0.0):
    a, h = self._get_address(x, s, return_head=True)
    m_bar = self._transform(x, s)
    if dropout > 0: m_bar = self.dropout(m_bar, dropout)
    return (1. - a) * gam + a * self._duplicate(m_bar), h

  def _read(self, gam, *inputs, head=None):
    if hub.gam_read_version == 1:
      a = self._get_address(*inputs, head=head)
      return self._summarize(a * gam)
    else:
      assert hub.gam_read_version == 0
      # Implement read operation basing on reshape
      # Runs slow but needs less RAM
      if head is None: head = self._get_head(*inputs)
      net_a = self.dense(self.total_size, head, self.address_scope)
      def operator(state, n_a):
        a = tf.nn.softmax(n_a, axis=1)
        return tf.reduce_sum(state * a, axis=1, keepdims=True)
      reshape2 = lambda _, n: n
      return self._binary_operate_over_groups(
        gam, net_a, operator, reshape2=reshape2)

  def _softmax_over_groups(self, tensor):
    """The 'softmax over groups' activation implemented using matrix
       multiplication
    """
    if hub.sog_version == 0: return super()._softmax_over_groups(tensor)
    assert hub.sog_version == 1
    assert isinstance(tensor, tf.Tensor) and len(tensor.shape) == 2
    # exp.shape = [M, SxN]
    exp = tf.exp(tensor)
    deno = self._duplicate(self._summarize(exp))
    return tf.divide(exp, deno, name='sog')

  # endregion : Operations V2

  # region : Private Methods

  def _reset_counter(self):
    """This method is used to avoid inappropriate increment by
       while-free build by recurrent base"""
    self._head_counter = 0
    self._address_counter = 0

  def _check_const_matrices(self):
    if self.D is None:
      if hub.sparse_gam: self._init_const_matrices_sp()
      else: self._init_const_matrices()

  @staticmethod
  def _matmul(x, y):
    if hub.sparse_gam: return tf.matrix_transpose(
        tf.sparse_tensor_dense_matmul(y, x, True, True))
    return tf.matmul(x, y)

  def _duplicate(self, tensor):
    self._check_const_matrices()
    assert isinstance(tensor, tf.Tensor)
    assert tensor.shape.as_list()[1] == self.num_groups
    return self._matmul(tensor, self.D)

  def _summarize(self, tensor):
    self._check_const_matrices()
    assert isinstance(tensor, tf.Tensor)
    assert tensor.shape.as_list()[1] == self.total_size
    return self._matmul(tensor, self.S)

  def _init_const_matrices(self):
    """M stands for batch_size, SxN is GAM config
    Duplicating:  Given X with shape (M, N), `D\in(N, SxN)`
                  Output Y with shape (M, SxN)
    Summarizing: Given X with shape (M, SxN), `S\in(SxN, N)`
                  Output Y with shape (M, N)
    """
    s, n = self.group_size, self.num_groups
    # Duplicating matrix D
    D = np.zeros((n, s * n), dtype=hub.np_dtype)
    indices=[[i, j] for i in range(n) for j in range(i*s, i*s+s)]
    for i, j in indices: D[i, j] = 1.0
    self.D = tf.constant(D, dtype=hub.dtype)
    # Summarizing matrix S
    S = np.transpose(D)
    self.S = tf.constant(S, dtype=hub.dtype)

  def _init_const_matrices_sp(self):
    """Initialize sparse matrices
    M stands for batch_size, SxN is GAM config
    Duplicating:  Given X with shape (M, N), `D\in(N, SxN)`
                  Output Y with shape (M, SxN)
    Summarizing: Given X with shape (M, SxN), `S\in(SxN, N)`
                  Output Y with shape (M, N)
    """
    s, n = self.group_size, self.num_groups
    # Duplicating matrix D
    self.D = tf.SparseTensor(
      indices=[[i, j] for i in range(n) for j in range(i*s, i*s+s)],
      values=[1.0] * self.total_size, dense_shape=[n, s*n])
    # Summarizing matrix S
    self.S = tf.SparseTensor(
      indices=[[i, j] for j in range(n) for i in range(j*s, j*s+s)],
      values=[1.0] * self.total_size, dense_shape=[s*n, n])

  # endregion : Private Methods

