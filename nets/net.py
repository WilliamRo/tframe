from __future__ import absolute_import

import six
import tensorflow as tf

from ..core import Function
from ..layers.layer import Layer
from ..layers.common import Activation
from ..utils import shape_string

from .. import pedia


class Net(Function):
  """Function which can packet sub-functions automatically when calling add
     method"""
  def __init__(self, name, f=None, level=0):
    """Instantiate Net, a name must be given"""
    self._name = name
    self._f = f
    self._level = level

    self.inputs = None
    self._last_scale = None

    self.chain = [f] if self.is_custom else []

  @property
  def group_name(self):
    return self._name

  @property
  def last_function(self):
    if len(self.chain) == 0:
      return None
    f = self.chain[-1]
    while isinstance(f, Net):
      f = f.chain[-1]

    return f

  @property
  def is_custom(self):
    return self._f is not None

  def _get_layer_string(self, f, scale):
    assert isinstance(f, Layer)
    result = ''

    result += f.abbreviation
    if scale and f.neuron_scale is not None:
      h_line = '_'
      for k in f.__dict__.keys():
        if 'regularizer' in k and f.__dict__[k] is not None:
          h_line = '-'
          break
      self._last_scale = shape_string(f.neuron_scale)
      result += '{}{}'.format(h_line, self._last_scale)

    return result

  def structure_string(self, detail=True, scale=True):
    fs = [f for f in self.chain if isinstance(f, Net)
          or detail or f.is_nucleus]
    assert isinstance(self.chain, list)
    result = ('' if self.inputs is None else 'input_{} => '.format(
      shape_string(self.inputs[0])))

    for (i, f) in zip(range(len(self.chain)), fs):
      if isinstance(f, Net):
        result += ' => ' if i != 0 else ''
        result += ('custom' if f.is_custom else
                   f.structure_string(detail, scale))
      else:
        assert isinstance(f, Layer)
        result += ' -> ' if i != 0 else ''
        result += self._get_layer_string(f, scale)

    if self._level == 0:
      result += ' => output_{}'.format(self.chain[-1]._last_scale)

    # Return
    return result

  @property
  def regularization_loss(self):
    reg_losses = tf.get_collection(pedia.tfkey.regularization_losses,
                                   self._name)
    loss_sum = None
    for loss in reg_losses:
      loss_sum = loss if loss_sum is None else loss_sum + loss
    return loss_sum

  def _link(self, inputs, **kwargs):
    # Check inputs
    if inputs is None:
      if self.inputs is None:
        raise ValueError('Input not defined')
      inputs = self.inputs

    # Check chain
    assert isinstance(self.chain, list)
    if len(self.chain) == 0:
      raise ValueError('Net is empty')

    with_logits = kwargs.get(pedia.with_logits, False)

    outputs = inputs
    logits = None
    # Link all functions in chain
    for f in self.chain:
      # Logits are always inputs of activation layers
      if isinstance(f, Activation):
        logits = outputs

      if isinstance(f, Net) and with_logits:
        outputs, logits = f(outputs, **kwargs)
      else:
        outputs = f(outputs)

      # Assign last_scale for custom net
      if isinstance(f, Net) and f.is_custom:
        f.last_scale = shape_string(outputs)

    # Return
    if with_logits:
      return outputs, logits
    else:
      return outputs

  def add(self, f):
    if isinstance(f, tf.Tensor):
      # If f is a placeholder
      if self.inputs is None:
        self.inputs = []
      self.inputs += [f]
      tf.add_to_collection(pedia.default_feed_dict, f)
    elif isinstance(f, Net) or self._level > 0:
      # Net should be added directly into self.chain
      self.chain += [f]
    elif isinstance(f, Layer):
      # If layer is a nucleus or the 1st layer added into this Net
      if f.is_nucleus or len(self.chain) == 0:
        self._wrap_and_add(f)
      # Otherwise add this layer to last Net of self.chain
      assert isinstance(self.chain[-1], Net)
      self.chain[-1].add(f)
    elif callable(f):
      self._wrap_and_add(f)
    else:
      raise ValueError('Object added to a Net must be a Layer or a Net or'
                        'callable')

  def _wrap_and_add(self, f):
    # Input f should be either a layer or a function
    assert callable(f)
    # Specify the name of the Net
    if isinstance(f, Layer) and not f.is_nucleus:
      name = 'Preprocess'
    else:
      name = f.abbreviation if isinstance(f, Layer) else 'Custom'
      index = 1
      get_name = lambda: '{}{}'.format(name, index)
      for f_ in self.chain:
        if isinstance(f_, Net) and f_.group_name == get_name():
          index += 1
      # Now we get the name
      name = get_name()

    # Wrap the layer into a new Net
    self.add(Net(name, f=(None if isinstance(f, Layer) else f),
                 level=self._level+1))


class Fork(Net):
  """Many to many net"""
  def __init__(self, name):
    Net.__init__(self, name, level=9)
    self.siblings = {}

  def structure_string(self, detail=True, scale=True):
    result = '('
    for i, key in zip(range(len(self.siblings)), self.siblings.keys()):
      f = self.siblings[key]
      assert isinstance(f, Layer)
      result += '' if i == 0 else ', '
      result += self._get_layer_string(f, scale)
    result += ')'

    self._last_scale = result
    return result

  def _link(self, inputs, **kwargs):
    with_logits = kwargs.get(pedia.with_logits, False)
    outputs = []
    for key in self.siblings.keys():
      f = self.siblings[key]
      assert isinstance(f, Layer)
      with tf.variable_scope(key):
        outputs.append(f(inputs))

    if with_logits:
      return outputs, None
    else:
      return outputs

  def add(self, name, f):
    if not isinstance(f, Layer):
      raise TypeError('Currently only Layer can be added to Fork')
    if not isinstance(name, six.string_types):
      raise TypeError('name must be a string')
    self.siblings[name] = f


