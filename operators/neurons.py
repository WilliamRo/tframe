from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub

from .psi_kernel import PsiKernel
from .bias_kernel import BiasKernel
from .apis.neurobase import NeuroBase


class NeuronArray(NeuroBase):

  def __init__(
      self,
      num_units,
      scope,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=False,
      bias_initializer='zeros',
      layer_normalization=False,
      **kwargs):

    # Call api's initializer
    NeuroBase.__init__(self, activation, weight_initializer, use_bias,
                       bias_initializer, layer_normalization, **kwargs)

    self.num_units = checker.check_positive_integer(num_units)
    self.scope = checker.check_type(scope, str)
    self.psi_kernels = []
    self.bias_kernel = None

  # region : Properties

  @property
  def etch(self):
    return self._nb_kwargs.get('etch', None)
  
  @property
  def being_etched(self):
    return self.etch is not None and hub.etch_on

  # endregion : Properties

  # region : Link

  def __call__(self, *input_list, **kwargs):
    """Link neuron array to graph
    :param input_list: inputs to be fully connected
    :return: neuron outputs
    """
    input_list = [x for x in input_list if x is not None]
    # Add inputs
    if input_list:
      checker.check_type(input_list, tf.Tensor)
      # Concatenation is forbidden when LN is on
      if all([len(input_list) > 1,
              self._layer_normalization,
              self._normalize_each_psi]):
        for x in input_list: self.add_kernel(x)
      else:
        # Concatenate to speed up calculation if necessary
        if len(input_list) > 1: x = tf.concat(input_list, axis=-1)
        else: x = input_list[0]
        # Add kernel
        self.add_kernel(x)

    # Make sure psi_kernel is not empty
    assert self.psi_kernels

    # Link
    with tf.variable_scope(self.scope):

      # Calculate summed input a, ref: Ba, etc. Layer Normalization, 2016
      a_list = [kernel() for kernel in self.psi_kernels]
      a = a_list[0] if len(a_list) == 1 else tf.add_n(a_list, 'summed_inputs')

      # Do layer normalization here if necessary
      if self._layer_normalization:
        if not self._normalize_each_psi:
          a = PsiKernel.layer_normalization(a, self._gain_initializer, False)
        # If LN is on, use_bias option must be True
        self._use_bias = True

      # Add bias if necessary
      if self._use_bias:
        if self.bias_kernel is None: self.register_bias_kernel()
        bias = self.bias_kernel()
        # Some kernels may generate bias of shape [batch_size, num_neurons]
        if len(bias.shape) == 1: a = tf.nn.bias_add(a, self.bias_kernel())
        else: a = a + bias

      # Activate if necessary
      if self._activation and kwargs.get('activate', True):
        a = self._activation(a)

    return a

  # endregion : Link

  # region : Public Methods

  def add_kernel(
      self, input_, kernel_key='fc', suffix=None, weight_initializer=None,
      prune_frac=0, **kwargs):
    """Add a psi kernel to self. Final neuron activation will be
       y = \phi(\Sigma_k(psi_k()) + bias)
    :param input_: psi input
    :param kernel_key: a string indicating which kernel should be called
                       during linking
    :param suffix: suffix of the variable scope of this kernel:
                   scope = 'psy_' + suffix.
    :param weight_initializer: if not provided, will be set to self's
    :param prune_frac: if positive, weights got inside kernel will be masked
                       and may be pruned in `lottery` style
    :param kwargs: additional arguments to call kernel, will be checked during
                   PsiKernel instantiating
    """

    if input_ is None: return
    # Set default suffix/weight_initializer if not provided
    if not suffix: suffix = str(len(self.psi_kernels) + 1)
    if weight_initializer is None: weight_initializer = self._weight_initializer

    # Initiate a psi_kernel
    psi_kernel = PsiKernel(
      kernel_key, self.num_units, input_, suffix,
      weight_initializer=weight_initializer, prune_frac=prune_frac,
      LN=self._layer_normalization and self._normalize_each_psi,
      gain_initializer=self._gain_initializer, etch=self.etch,
      weight_dropout=self._weight_dropout, **kwargs)

    self.psi_kernels.append(psi_kernel)

  def register_bias_kernel(self, kernel_key='common', prune_frac=0., **kwargs):
    self.bias_kernel = BiasKernel(
      kernel_key, self.num_units, self._bias_initializer,
      prune_frac, **kwargs)

  # endregion : Public Methods

