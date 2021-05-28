from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import hub
from tframe import linker
from tframe import activations


class GenericNeurons(object):

  @staticmethod
  def eta(num, activation, *tensors, weight_initializer='glorot_normal',
          use_bias=True, bias_initializer='zeros'):
    """
    * should be used inside a variable scope
    """
    a = psi(num, *tensors, weight_initializer, use_bias, bias_initializer)
    if activation is not None:
      activation = activations.get(activation)
      a = activation(a)
    return a

  @staticmethod
  def psi(num, *tensors, weight_initializer='glorot_normal', use_bias=True,
          bias_initializer='zeros', additional_summed_input=None, suffix=''):
    """
    * should be used inside a variable scope
    """
    assert len(tensors) > 0
    if len(tensors) == 1: x = tensors[0]
    else: x = tf.concat(tensors, axis=1)

    # Calculate summed input
    input_dim = linker.get_dimension(x)
    W = tf.get_variable('W' + suffix, shape=[input_dim, num], dtype=hub.dtype,
                        initializer=weight_initializer)
    a = x @ W

    # Add additional summed input if provided
    if additional_summed_input is not None: a += additional_summed_input
    # Add bias if necessary
    if use_bias: a = add_bias(a, bias_initializer)

    return a

  @staticmethod
  def psi_k(num, *tensors, weight_initializer='glorot_normal', suffix=''):
    return psi(num, *tensors, weight_initializer=weight_initializer,
               use_bias=False, suffix=suffix)

  @staticmethod
  def add_bias(a, bias_initializer):
    num = linker.get_dimension(a)
    bias = tf.get_variable('bias', shape=[num], dtype=hub.dtype,
                           initializer=bias_initializer)
    return tf.nn.bias_add(a, bias)


eta = GenericNeurons.eta
psi = GenericNeurons.psi
psi_k = GenericNeurons.psi_k
add_bias = GenericNeurons.add_bias
