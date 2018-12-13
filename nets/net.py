from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.core import Function
from tframe import context
from tframe.layers.layer import Layer
from tframe.layers import Activation
from tframe.layers import Input
from tframe.utils import shape_string
import tframe.utils.format_string as fs

from tframe import pedia
from tframe.core.decorators import with_graph_if_has


class Net(Function):
  """Function which can packet sub-functions automatically when calling add
     method"""

  CASCADE = pedia.cascade
  PROD = pedia.prod
  SUM = pedia.sum
  FORK = pedia.fork
  CONCAT = pedia.concat
  RECURRENT = 'RECURRENT'

  def __init__(self, name, level=0, inter_type=pedia.cascade,
               is_branch=False, **kwargs):
    """Instantiate Net, a name must be given
       :param level: level 0 indicates the trunk
       :param inter_type: \in {cascade, fork, sum, prod, concat}
    """
    self.name = name
    self._level = level
    self._inter_type = inter_type
    self.is_branch = is_branch

    self.input_ = None
    self._output_scale = None

    self.children = []
    self.branch_outputs = []
    self._kwargs = kwargs

    self._logits_tensor = None

    # Losses
    self._extra_loss = None
    self._reg_loss = None
    self._customized_loss = None

  # region : Properties

  @property
  def var_list(self):
    """Should be used in with graph context"""
    return [var for var in tf.trainable_variables()
            if '{}'.format(self.name) == var.name.split('/')[self._level]]

  @property
  def weight_list(self):
    return [var for var in self.var_list if 'weights' in var.name]

  @property
  def params_num(self):
    return sum([np.prod(param.shape) for param in self.var_list])

  @property
  def group_name(self):
    return self.name

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
  
  @property
  @with_graph_if_has
  def structure_detail(self):
    """A list of structure strings with format
       Layer (type)           Output Shape           Params #
    Currently only work for sequential model
    """
    from tframe.nets.rnet import RNet
    widths = [50, 16]
    indent = 3

    rows = []
    total_params = 0
    if self.is_root:
      input_info = 'Input_{}'.format(shape_string(self.input_.sample_shape))
      rows.append(fs.table_row([input_info, ''], widths))
    for child in self.children:
      if isinstance(child, Layer):
        # Try to find variable in child
        tensors = [v for v in self.var_list if child.group_name in v.name]
        num = sum([np.prod(t.shape) for t in tensors])
        num_str = '' if num == 0 else '{}'.format(num)
        cols = [self._get_layer_string(child, True, True), num_str]

        rows.append(fs.table_row(cols, widths))
        total_params += num
      elif isinstance(child, RNet):
        num = child.params_num
        cols = [child.structure_string(), '{}'.format(num)]

        rows.append(fs.table_row(cols, widths))
        total_params += num
      elif isinstance(child, Net):
        _rows, _total_params = child.structure_detail
        rows += _rows
        total_params += _total_params
      else:
        raise TypeError('!! unknown child type {}'.format(type(child)))

    if self.is_root:
      # Head
      detail = ''
      add_with_indent = lambda d, c: d + ' ' * indent + c + '\n'
      width = sum(widths)
      detail = add_with_indent(detail, '-' * width)
      detail = add_with_indent(
        detail, fs.table_row(['Layers', 'Params #'], widths))
      detail = add_with_indent(detail, '=' * width)
      # Content
      for i, row in enumerate(rows):
        if i > 0:
          detail = add_with_indent(detail, '-' * width)
        detail = add_with_indent(detail, row)
      # Summary
      detail = add_with_indent(detail, '=' * width)
      detail = add_with_indent(
        detail, 'Total params: {}'.format(total_params))
      detail += ' ' * indent + '-' * width
      return detail, total_params
    else: return rows, total_params

  def _get_layer_string(self, f, scale, full_name=False):
    assert isinstance(f, Layer)
    result = f.abbreviation if not full_name else f.full_name
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
    if self._inter_type not in (pedia.cascade,
                                self.RECURRENT) or self.is_branch:
      if self._inter_type in [pedia.sum, pedia.prod, pedia.concat]:
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
    if self._inter_type not in (pedia.cascade,
                                self.RECURRENT) or self.is_branch: result += ')'
    # Add output scale
    if self.is_root and not self._inter_type == pedia.fork:
      result += ' => output_{}'.format(self.children[-1]._output_scale)

    # Return
    return result

  @property
  def regularization_loss(self):
    if self._reg_loss is not None: return self._reg_loss

    reg_losses = tf.get_collection(
      pedia.tfkey.regularization_losses, self.name)
    if len(reg_losses) > 0:
      self._reg_loss = tf.add_n(reg_losses, name='reg_sum')
    return self._reg_loss

  @property
  def customized_loss(self):
    if self._customized_loss is not None: return self._customized_loss

    f = context.read_value(pedia.custom_loss_f, default_value=None)
    if callable(f): self._customized_loss = f(self)

    if self._customized_loss is not None:
      assert isinstance(self._customized_loss, tf.Tensor)
    return self._customized_loss

  @property
  def extra_loss(self):
    if self._extra_loss is not None: return self._extra_loss
    add = lambda v: v if self._extra_loss is None else self._extra_loss + v

    if self.regularization_loss is not None:
      self._extra_loss = add(self.regularization_loss)

    if self.customized_loss is not None:
      self._extra_loss = add(self.customized_loss)

    return self._extra_loss

  # endregion : Properties


  # region : Overrode Method

  # TODO: modify with_logits mechanism
  def _link(self, *inputs, **kwargs):
    # region : Check inputs

    if len(inputs) == 0 or inputs[0] is None:
      input_ = self.input_() if self.input_ is not None else None
    elif len(inputs) == 1: input_ = inputs[0]
    else: raise SyntaxError('!! Too much inputs')

    if input_ is not None and not isinstance(input_, tf.Tensor):
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
    if self._inter_type == self.FORK:
      output = output_list
      self.branch_outputs = output
    elif self._inter_type == self.SUM:
      output = tf.add_n(output_list)
    elif self._inter_type == self.PROD:
      output = output_list.pop()
      for tensor in output_list: output *= tensor
    elif self._inter_type == self.CONCAT:
      output = tf.concat(output_list, axis=-1)
    elif self._inter_type != self.CASCADE:
      raise TypeError('!! Unknown net inter type {}'.format(self._inter_type))

    return output

  # endregion : Overrode Methods


  # region : Public Methods

  def pop_last_softmax(self):
    """For RNN classifiers, softmax layer should be applied outside
       while-loop"""
    assert self.is_root
    if not (isinstance(self.last_function, Activation)
        and self.last_function.abbreviation == 'softmax'):
      return None

    last_net = self.children[-1]
    assert isinstance(last_net, Net) and len(last_net.children) > 0
    last_layer = last_net.children[-1]
    assert (isinstance(last_layer, Activation)
            and last_layer.abbreviation == 'softmax')
    layer = last_net.children.pop(-1)

    if len(last_net.children) == 0: self.children.pop(-1)
    return layer

  def add_to_last_net(self, layer, only_cascade=False):
    from tframe.nets.rnet import RNet

    if len(self.children) == 0:
      raise AssertionError('!! This net does not have children')
    last_net = self.children[-1]
    if isinstance(last_net, RNet) or (only_cascade and
                                      last_net._inter_type != self.CASCADE):
      last_net = self._add_new_subnet(layer)

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
          self._inter_type not in (self.CASCADE, self.RECURRENT)):
      # Net should be added directly into self.children of any net
      # Layer should be added directly into self.children for non-cascade nets
      return self._save_add(f)
    elif isinstance(f, Layer):
      # If layer is a nucleus or the 1st layer added into this Net
      if f.is_nucleus or len(self.children) == 0:
        self._add_new_subnet(f)
      # Otherwise add this layer to last Net of self.children
      return self.add_to_last_net(f, only_cascade=True)
    else: raise ValueError('!! Object added to a Net must be a Layer or a Net')

  # endregion : Public Methods


  # region : Private Methods

  def _save_add(self, f):
    # TODO: avoid name scope conflict when add layers to non-cascade nets
    name = self._get_new_name(f)
    net = self
    if isinstance(f, Layer): f.full_name = name
    elif isinstance(f, Net):
      f._level = self._level + 1
      f.name = name
      net = f
    self.children.append(f)
    return net

  def _add_new_subnet(self, layer):
    # Input f should be a layer
    assert isinstance(layer, Layer)
    # Specify the name of the Net
    if len(self.children) == 0: name = 'Preprocess'
    else: name = self._get_new_name(layer.abbreviation)

    # Wrap the layer into a new Net
    return self.add(Net(name, level=self._level + 1))

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
