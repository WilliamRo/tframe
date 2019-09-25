from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import console
from tframe.core import Function
from tframe import context
from tframe.layers.layer import Layer
from tframe.layers import Input
from tframe.utils import shape_string
import tframe.utils.format_string as fs
from tframe.utils.string_tools import merger

from tframe import pedia
from tframe import hub
from tframe.utils import stark
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
    self.kwargs = kwargs

    # Losses
    self._extra_loss = None
    # self._reg_loss = None

    # Tensor extractor
    self._tensor_extractors = []

  # region : Properties

  @property
  def var_list(self):
    """Should be used in with graph context"""
    return [var for var in tf.trainable_variables()
            if '{}'.format(self.name) == var.name.split('/')[self._level]]

  @property
  def weight_vars(self):
    vars = []
    for v in self.var_list:
      assert isinstance(v, tf.Variable)
      name = v.name.split('/')[-1]
      if 'w' == name.lower()[0]: vars.append(v)
    return vars

  @property
  def weight_list(self):
    return [var for var in self.var_list if 'weights' in var.name]

  @property
  def params_num(self):
    return stark.get_params_num(self.var_list, consider_prune=True)

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
    return context.logits_tensor

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
    widths = [33, 24, 20]
    indent = 3

    rows = []
    add_to_rows = lambda cols: rows.append(fs.table_row(cols, widths))
    # Dense total will be used when model weights are pruned
    total_params, dense_total = 0, 0
    if self.is_root:
      add_to_rows(['input', shape_string(self.input_.sample_shape), ''])

    def get_num_string(num, dense_num):
      if num == 0: num_str = ''
      elif hub.prune_on or hub.etch_on:
        num_str = '{} ({:.1f}%)'.format(num, 100.0 * num / dense_num)
      else: num_str = str(num)
      return num_str

    for child in self.children:
      if isinstance(child, Layer):
        # Try to find variable in child
        variables = [v for v in self.var_list if child.group_name in v.name]
        num, dense_num = stark.get_params_num(variables, consider_prune=True)
        # Generate a row
        cols = [self._get_layer_string(child, True, True),
                child.output_shape_str, get_num_string(num, dense_num)]
        add_to_rows(cols)
      elif isinstance(child, RNet):
        num, dense_num = child.params_num
        cols = [child.structure_string(), child.output_shape_str,
                get_num_string(num, dense_num)]
        add_to_rows(cols)
      elif isinstance(child, Net):
        _rows, num, dense_num = child.structure_detail
        rows += _rows
      else:
        raise TypeError('!! unknown child type {}'.format(type(child)))

      # Accumulate total_params and dense_total_params
      total_params += num
      dense_total += dense_num

    if self.is_root:
      # Head
      detail = ''
      add_with_indent = lambda d, c: d + ' ' * indent + c + '\n'
      width = sum(widths)
      detail = add_with_indent(detail, '-' * width)
      detail = add_with_indent(
        detail, fs.table_row(['Layers', 'Output Shape', 'Params #'], widths))
      detail = add_with_indent(detail, '=' * width)
      # Content
      for i, row in enumerate(rows):
        if i > 0:
          detail = add_with_indent(detail, '-' * width)
        detail = add_with_indent(detail, row)
      # Summary
      detail = add_with_indent(detail, '=' * width)
      detail = add_with_indent(
        detail, 'Total params: {}'.format(
          get_num_string(total_params, dense_total)))
      detail += ' ' * indent + '-' * width
      return detail, total_params, dense_total
    else: return rows, total_params, dense_total

  def _get_layer_string(self, f, scale, full_name=False):
    assert isinstance(f, Layer)
    return f.get_layer_string(scale, full_name)

  def structure_string(self, detail=True, scale=True):
    # Get functions to be added to structure string
    assert isinstance(self.children, list)
    fs = [f for f in self.children if isinstance(f, Net)
          or detail or f.is_nucleus]

    # Add input layer
    result = ('' if self.input_ is None else 'input[{}] => '.format(
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
    str_list, next_token = [], None
    for f in fs:
      if isinstance(f, Net):
        if next_token is None: next_token = next_net
        assert next_token == next_net
        str_list.append(f.structure_string(detail, scale))
      else:
        assert isinstance(f, Layer)
        if next_token is None: next_token = next_layer
        assert next_token == next_layer
        str_list.append(self._get_layer_string(f, scale))

    str_list = merger(str_list)
    result += next_token.join(str_list)

    # Check is_branch flag
    if self.is_branch:
      result += ' -> output'

    # Check interconnection type
    if self._inter_type not in (pedia.cascade,
                                self.RECURRENT) or self.is_branch: result += ')'
    # Add output scale
    if self.is_root and not self._inter_type == pedia.fork:
      result += ' => output[{}]'.format(self.output_shape_str)

    # Return
    return result

  @property
  def extra_loss(self):
    """When this property is accessed for the 1st time in model.build:
        For RNN, self._extra_loss has already been calculated
        For FNN, self._extra_loss is None, and needed to be calculated
    """
    if self._extra_loss is None: self._extra_loss = self._get_extra_loss()
    return self._extra_loss

  @property
  def layers(self):
    """A customized net is also took as layer"""
    if len(self.children) == 0: return [self]
    layers = []
    for child in self.children:
      if isinstance(child, Layer): layers.append(child)
      else:
        assert isinstance(child, Net)
        layers += child.layers
    return layers

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
    # if len(self.children) == 0: raise ValueError('!! Net is empty')

    pioneer = input_
    output_list = []
    output = None
    # Link all functions in children
    for f in self.children:
      # Handle branches
      if isinstance(f, Net) and f.is_branch:
        self.branch_outputs.append(f(pioneer))
        continue

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

    # This will only happens when Net is empty
    if output is None: output = input_

    # Extract tensors to export
    if self.is_root:
      # Run customized extractor
      for extractor in self._tensor_extractors:
        assert callable(extractor)
        extractor(self)
      # Run build-in extractors
      self.variable_extractor()

    # Return
    return output

  # endregion : Overrode Methods


  # region : Public Methods

  def register_extractor(self, extractor):
    """Extractors will be used to extract tensors to export while linking"""
    assert callable(extractor)
    self._tensor_extractors.append(extractor)

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

  def add(self, f=None, inter_type=pedia.cascade, return_net=False):
    """Add a net or a layer in to this model
    :param f: \in (Net, Layer)
    :param inter_type: inter-connection type
    :return: f or f's container
    """
    # If add an empty net
    if f is None:
      name = self._get_new_name(inter_type)
      net = Net(name, level=self._level + 1, inter_type=inter_type)
      self.children.append(net)
      return net

    # If add a function to this net
    container = self
    if isinstance(f, Input):
      # If f is a placeholder
      self.input_ = f
    elif (isinstance(f, Net) or not self.is_root or
          self._inter_type not in (self.CASCADE, self.RECURRENT)):
      # Net should be added directly into self.children of any net
      # Layer should be added directly into self.children for non-cascade nets
      container = self._save_add(f)
    elif isinstance(f, Layer):
      # If layer is a nucleus or the 1st layer added into this Net
      if f.is_nucleus or len(self.children) == 0:
        self._add_new_subnet(f)
      # Otherwise add this layer to last Net of self.children
      container = self.add_to_last_net(f, only_cascade=True)
    else: raise ValueError('!! Object added to a Net must be a Layer or a Net')

    if return_net: return container
    else: return f

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
    # if len(self.children) == 0: name = 'Preprocess'
    if len(self.children) == 0 and not layer.is_nucleus: name = 'Preprocess'
    else: name = self._get_new_name(layer.abbreviation)

    # Wrap the layer into a new Net
    return self.add(Net(name, level=self._level + 1), return_net=True)

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

  def _get_customized_loss(self, outer=False):
    f = (context.customed_outer_loss_f_net if outer
         else context.customed_loss_f_net)
    if callable(f):
      loss_list = f(self)
      assert isinstance(loss_list, list)
      return loss_list
    else: return []

  def _get_extra_loss(self):
    loss_tensor_list = context.loss_tensor_list
    assert isinstance(loss_tensor_list, list)
    customized_loss = self._get_customized_loss()
    if customized_loss:
      loss_tensor_list += customized_loss
    # Add regularizer losses
    loss_tensor_list += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Add loss tensor list
    if loss_tensor_list:
      result = tf.add_n(loss_tensor_list, 'extra_loss')
    else: result = None

    # Show loss list
    if hub.show_extra_loss_info and loss_tensor_list:
      console.show_info('Extra losses:')
      for loss_tensor in loss_tensor_list:
        assert isinstance(loss_tensor, tf.Tensor)
        console.supplement(loss_tensor.name, level=2)
      console.split()
    return result

  # endregion: Private Methods

  # region : Link tools

  def _get_variable(self, name, shape, initializer=None):
    if initializer is None:
      initializer = getattr(
        self, '_weight_initializer', tf.glorot_normal_initializer())
    else:
      assert callable(initializer)
    return tf.get_variable(
      name, shape, dtype=hub.dtype, initializer=initializer)

  def _get_bias(self, name, dim, initializer=None):
    if initializer is None:
      initializer = getattr(self, '_bias_initializer', tf.zeros_initializer)
    else:
      assert callable(initializer)
    return tf.get_variable(
      name, shape=[dim], dtype=hub.dtype, initializer=initializer)

  @staticmethod
  def _get_shape_list(tensor):
    assert isinstance(tensor, tf.Tensor)
    return tensor.shape.as_list()

  # endregion : Link tools

  # region : Overrides

  def __str__(self):
    return self.structure_string()

  def __repr__(self):
    return self.structure_string()

  # endregion : Overrides

  # region : Build-in extractors

  def variable_extractor(self):
    get_key = lambda v: '/'.join(v.name.split('/')[1:])
    def add_to_dict(v): context.variables_to_export[get_key(v)] = v

    if hub.export_weights:
      for v in self.weight_vars: add_to_dict(v)

    # if hub.export_masked_weights and hub.pruning_rate_fc > 0:
    if hub.export_masked_weights:
      from tframe.operators.prune.pruner import Pruner
      Pruner.extractor()

    if hub.export_sparse_weights:
      for v in context.sparse_weights_list:
        assert isinstance(v, (tf.Tensor, tf.Variable))
        # TODO: temporal solution to circumvent conflicts
        if 'scan' in v.name.lower(): continue
        add_to_dict(v)

    # Register weights
    self._register_weights_to_monitor()

  def _register_weights_to_monitor(self):
    """<monitor_grad_step_01: register to monitor>"""
    if not hub.monitor_weight_grads: return
    monitor = context.monitor

    # weights of type tf.Variable
    monitor.register_weights(self.weight_vars)

    # TODO: register masked_weights and sparse weights

  # endregion : Build-in extractors
