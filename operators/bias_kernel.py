from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import hub
from tframe import linker
from tframe import initializers

from .kernel_base import KernelBase


class BiasKernel(KernelBase):

  # region : Public Methods

  def __call__(self):
    bias = self.kernel(**self.kwargs)
    return bias

  # endregion : Public Methods

  # region : Private Methods

  def _get_kernel(self, identifier):
    assert isinstance(identifier, str)
    identifier = identifier.lower()
    if identifier in ('common', 'traditional'): return self.common
    elif identifier in ('hyper', 'hyper16'): return self.hyper16
    else: raise ValueError('!! Unknown kernel `{}`'.format(identifier))

  # endregion : Private Methods

  # region : Kernels

  def common(self):
    bias = tf.get_variable('bias', shape=[self.num_units], dtype=hub.dtype,
                           initializer=self.initializer)
    return bias

  def hyper16(self, seed, weight_initializer):
    shape = [linker.get_dimension(seed), self.num_units]
    weight_initializer = initializers.get(weight_initializer)
    Wzb = self._get_weights('Wzb', shape, initializer=weight_initializer)

    bias = seed @ Wzb
    b0 = tf.get_variable('bias', shape=[self.num_units], dtype=hub.dtype,
                         initializer=self.initializer)
    return tf.nn.bias_add(bias, b0)

  # endregion : Kernels
