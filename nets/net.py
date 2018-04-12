from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tframe.core import Function
from tframe.layers.layer import Layer
from tframe.layers import Activation
from tframe.layers import Input
from tframe.utils import shape_string

from tframe import pedia


class Net(Function):
  """Function which can packet sub-functions automatically when calling add
     method"""
  def __init__(self, name, level=0, inter_type=pedia.cascade,
               is_branch=False, **kwargs):
    """Instantiate Net, a name must be given
       :param level: level 0 indicates the trunk
       :param inter_type: \in {cascade, fork, sum, prod}
    """
    self._name = name
    self._level = level
    self._inter_type = inter_type
    self.is_branch = is_branch

    self.input_ = None
    self._output_scale = None

    self.children = []
    self.branch_outputs = []
    self._kwargs = kwargs

    self._logits_tensor = None


  # region : Properties

  @property
  def var_list(self):
    return [var for var in tf.trainable_variables()
            if '{}'.format(self._name) == var.name.split('/')[self._level]]

  @property
  def group_name(self):
    return self._name

  @property
  def last_function(self):
    if len(self.children) == 0 or self._inter_type in (pedia.prod, pedia.sum):
      return None
    f = self.children[-1]
    while isinstance(f, Net): f = f.last_function
    return f

  @property
  def input_tensor(self):
    if self.input_ is None: raise ValueError('!! Input not found')
    return self.input_.place_holder

  @property
  def logits_tensor(self):
    if self._logits_tensor is not None: return self._logits_tensor
    for child in reversed(self.children):
      if isinstance(child, Net) and child.logits_tensor is not None:
        return child.logits_tensor
    return None

  @property
  def is_root(self):
    return self._level == 0

  def _get_layer_string(self, f, scale):
    assert isinstance(f, Layer)
    result = f.abbreviation
    if scale and f.neuron_scale is not None:
      self._output_scale = shape_string(
        f.neuron_scale if f.output_scale is None else f.output_scale)
      result += '_{}'.format(f.neuron_scale if len(f.neuron_scale) > 1 else
                             f.neuron_scale[0])
    return result

  def structure_string(self, detail=True, scale=True):
    # Get functions to be added to structure string
    assert isinstance(self.children, list)
    fs = [f for f in self.children if isinstance(f, Net)
          or detail or f.is_nucleus]

    # Add input layer
    result = ('' if self.input_ is None else 'input_{} => '.format(
      shape_string(self.input_.sample_shape)))

    # Check interconnection type
    next_net, next_layer = ' => ', ' -> '
    if self._inter_type != pedia.cascade or self.is_branch:
      if self._inter_type in [pedia.sum, pedia.prod]:
        result += self._inter_type
      if self.is_branch: result += 'branch'
      else: next_layer, next_net = ', ', ', '
      result += '('

    # Add children
    for (i, f) in zip(range(len(self.children)), fs):
      if isinstance(f, Net):
        result += next_net if i != 0 else ''
        result += f.structure_string(detail, scale)
      else:
        assert isinstance(f, Layer)
        result += next_layer if i != 0 else ''
        result += self._get_layer_string(f, scale)

    # Check is_branch flag
    if self.is_branch:
      result += ' -> output'

    # Check interconnection type
    if self._inter_type != pedia.cascade or self.is_branch: result += ')'

    # Add output scale
    if self.is_root:
      result += ' => output_{}'.format(self.children[-1]._output_scale)

    # Return
    return result

  @property
  def regularization_loss(self):
    reg_losses = tf.get_collection(
      pedia.tfkey.regularization_losses, self._name)
    return (None if len(reg_losses) == 0
            else tf.add_n(reg_losses, name='reg_sum'))

  # endregion : Properties


  # region : Overrode Method

  # TODO: modify with_logits mechanism
  def _link(self, *inputs, **kwargs):
    # region : Check inputs
    if len(inputs) == 0:
      if self.input_ is None: raise ValueError('!! Input not defined')
      input_ = self.input_()
    elif len(inputs) == 1: input_ = inputs[0]
    else: raise SyntaxError('!! Too much inputs')
    if not isinstance(input_, tf.Tensor):
      raise TypeError('!! input should be a Tensor')
    # endregion : Check inputs

    # Check children
    assert isinstance(self.children, list)
    if len(self.children) == 0: raise ValueError('!! Net is empty')

    pioneer = input_
    output_list = []
    output = None
    # Link all functions in children
    for f in self.children:
      # Handle branches
      if isinstance(f, Net) and f.is_branch:
        self.branch_outputs.append(f(pioneer))
        continue

      # Logits are always inputs of activation layers
      if isinstance(f, Activation): self._logits_tensor = pioneer

      # Call each child
      output = f(pioneer)

      if self._inter_type == pedia.cascade: pioneer = output
      else: output_list.append(output)

    # Calculate output
    if self._inter_type == pedia.fork: output = output_list
    elif self._inter_type == pedia.sum: output = tf.add_n(output_list)
    elif self._inter_type == pedia.prod:
      output = output_list.pop()
      for tensor in output_list: output *= tensor

    # Return
    return output

  # endregion : Overrode Methods


  # region : Public Methods

  def add_to_last_net(self, layer):
    if len(self.children) == 0:
      raise AssertionError('!! This net does not have children')
    last_net = self.children[-1]
    assert isinstance(last_net, Net)
    last_net.add(layer)
    return last_net

  def add_branch(self):
    if not self.is_root: raise ValueError('Branches can only added to the root')
    net = Net(name='branch', is_branch=True)
    self.add(net)

    return net

  def add(self, f=None, inter_type=pedia.cascade):
    # If add an empty net
    if f is None:
      name = self._get_new_name(inter_type)
      net = Net(name, level=self._level + 1, inter_type=inter_type)
      self.children.append(net)
      return net

    # If add a function to this net
    if isinstance(f, Input):
      # If f is a placeholder
      self.input_ = f
      return self
    elif (isinstance(f, Net) or not self.is_root or
           self._inter_type != pedia.cascade):
      # Net should be added directly into self.children of any net
      # Layer should be added directly into self.children for non-cascade nets
      return self._save_add(f)
    elif isinstance(f, Layer):
      # If layer is a nucleus or the 1st layer added into this Net
      if f.is_nucleus or len(self.children) == 0: self._wrap_and_add(f)
      # Otherwise add this layer to last Net of self.children
      return self.add_to_last_net(f)
    else: raise ValueError(
      'Object added to a Net must be a Layer or a Net')

  # endregion : Public Methods


  # region : Private Methods

  def _save_add(self, f):
    # TODO: avoid name scope conflict when add layers to non-cascade nets
    name = self._get_new_name(f)
    net = self
    if isinstance(f, Layer): f.full_name = name
    elif isinstance(f, Net):
      f._level = self._level + 1
      f._name = name
      net = f
    self.children.append(f)
    return net

  def _wrap_and_add(self, layer):
    # Input f should be a layer
    assert isinstance(layer, Layer)
    # Specify the name of the Net
    if not layer.is_nucleus: name = 'Preprocess'
    else: name = self._get_new_name(layer.abbreviation)

    # Wrap the layer into a new Net
    self.add(Net(name, level=self._level + 1))

  def _get_new_name(self, entity):
    if isinstance(entity, Net): name = entity.group_name
    elif isinstance(entity, Layer): name = entity.full_name
    else: name = entity
    index = 1
    get_name = lambda: '{}{}'.format(name, '' if index == 1 else index)

    for f_ in self.children:
      if isinstance(entity, Layer) and isinstance(f_, Layer):
        if f_.full_name == get_name(): index += 1
      elif f_.group_name == get_name(): index += 1

    return get_name()

  # endregion: Private Methods


# region : Deprecated

# TODO: deprecate class Fork
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
    outputs = []
    for key in self.siblings.keys():
      f = self.siblings[key]
      assert isinstance(f, Layer)
      with tf.variable_scope(key):
        outputs.append(f(inputs))

      return outputs

  def add(self, name, f):
    if not isinstance(f, Layer):
      raise TypeError('Currently only Layer can be added to Fork')
    if not isinstance(name, six.string_types):
      raise TypeError('name must be a string')
    self.siblings[name] = f

# endregion : Deprecated
