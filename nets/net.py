from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import console
from tframe import context
from tframe import hub
from tframe import pedia
from tframe.core import Function
from tframe.core import Nomear
from tframe.core.decorators import with_graph_if_has
from tframe.core.slots import OutputSlot
from tframe.layers.layer import Layer
from tframe.layers import Input
from tframe.utils import shape_string
from tframe.utils import stark
from tframe.utils.display.table import Table
from tframe.utils.string_tools import merger



class Net(Function, Nomear):
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
       TODO: deprecate inter_type
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

    self._output_slots = []

  # region : Properties

  @property
  def var_list(self):
    """Should be used in with graph context"""
    return [var for var in tf.trainable_variables()
            if '{}'.format(self.name) == var.name.split('/')[self._level]]

  @property
  def decayable_vars(self):
    return [var for var in self.var_list if stark.decayable(var)]

  @property
  def weight_vars(self):
    vars = []
    for v in self.var_list:
      assert isinstance(v, tf.Variable)
      name = v.name.split('/')[-1]
      if name.lower().split(':')[0] in ('kernel', 'w', 'weight'):
        vars.append(v)
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
  def output_slots(self):
    results = self._output_slots
    for child in self.children:
      if isinstance(child, Net): results += child.output_slots
    assert isinstance(results, list)
    return results

  @property
  def input_tensor(self):
    if self.input_ is None: raise ValueError('!! Input not found')
    return self.input_.place_holder

  @property
  def logits_tensor(self):
    """This property should be visited only by RNNs"""
    tensors = list(context.logits_tensor_dict.values())
    if len(tensors) == 0: return None
    assert len(tensors) == 1
    return tensors[0]

  @property
  def is_root(self):
    return self._level == 0
  
  @property
  @with_graph_if_has
  def structure_detail(self):
    """A list of structure strings with format
       Layer (type)           Output Shape           Params #
    Currently only work for sequential model
    TODO: refactoring is badly needed
    """
    from tframe.nets.rnet import RNet
    from tframe.nets.customized_net import CustomizedNet
    indent = 3

    # rows is a list of lists of 3 cols
    rows = []

    # Dense total will be used when model weights are pruned
    total_params, dense_total = 0, 0
    if self.input_ is not None:
      rows.append(['input', shape_string(self.input_.sample_shape), ''])

    for child in self.children:
      if isinstance(child, Layer):
        _row, num, dense_num = self._get_layer_detail(child)
        rows.append(_row)
      elif isinstance(child, (RNet, CustomizedNet)):
        num, dense_num = child.params_num
        cols = [child.structure_string(), child.output_shape_str,
                stark.get_num_string(num, dense_num)]
        rows.append(cols)
      elif isinstance(child, Net):
        _rows, num, dense_num = child.structure_detail
        # TODO
        rows += _rows
      else:
        raise TypeError('!! unknown child type {}'.format(type(child)))

      # Accumulate total_params and dense_total_params
      total_params += num
      dense_total += dense_num

    # Check total params
    if not (hub.prune_on or hub.etch_on):
      var_list_params = sum([np.prod(v.shape) for v in self.var_list])
      if not total_params == var_list_params:
        raise AssertionError('!! total params do not match')

    if self.is_root:
      headers = ['Layers', 'Output Shape', 'Params #']
      # Decide cell widths
      widths = [max(len(h), max([len(r[i]) for r in rows]))
                for i, h in enumerate(headers)]
      # Put all these stuff into a table
      t = Table(*widths, margin=0, tab=9, buffered=True, indent=indent)
      t.specify_format(align='llr')
      t.print_header(*headers)
      for i, row in enumerate(rows):
        # Replace unknown shape with `?`. This takes effect in models with
        #  input of partially unknown shape
        row[1] = row[1].replace('None', '?')

        t.print_row(*row)
        # Draw line
        t.hline() if i != len(rows) - 1 else t.dhline()
      t.print_with_margin('Total params: {}'.format(
        stark.get_num_string(total_params, dense_total)))
      t.hline()
      return t.content, total_params, dense_total
    else: return rows, total_params, dense_total

  def _get_layer_detail(self, layer, suffix=''):
    variables = [v for v in self.var_list
                 if layer.group_name == v.name.split('/')[self._level + 1]]
    num, dense_num = stark.get_params_num(variables, consider_prune=True)
    # Generate a row
    row = [self._get_layer_string(layer, True, True, suffix),
           layer.output_shape_str, stark.get_num_string(num, dense_num)]
    return row, num, dense_num

  def _get_layer_string(self, f, scale, full_name=False, suffix=''):
    assert isinstance(f, Layer)
    return f.get_layer_string(scale, full_name, suffix)

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
    if self.is_branch: result += 'branch('

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
    if self.is_branch: result += ' -> output)'

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

  @Nomear.property()
  def weight_clip_ops(self):
    value = hub.clip_weight_at
    assert isinstance(value, float) and value > 0
    ops = [tf.assign(v, tf.clip_by_value(v, -value, value))
           for v in self.weight_vars]
    return ops

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
      if self.is_root and hub.export_activations:
        context.monitor.register_tensor(
          output, reduce_1st_dim=True, name='activation')

      if self._inter_type == pedia.cascade: pioneer = output
      else: output_list.append(output)

    # Calculate output
    if self._inter_type == self.FORK:
      output = output_list
      self.branch_outputs = output
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

    if len(self.children) == 0:
      raise AssertionError('!! This net does not have children')
    last_net = self.children[-1]
    if type(last_net) is not Net:
    # if isinstance(last_net, RNet) or (only_cascade and
    #                                   last_net._inter_type != self.CASCADE):
      last_net = self._add_new_subnet(layer)

    assert isinstance(last_net, Net)
    last_net.add(layer)
    return last_net

  def add_branch(self):
    if not self.is_root: raise ValueError('Branches can only added to the root')
    net = Net(name='branch', is_branch=True)
    self.add(net)
    return net

  def add_forkmerge(self, merge_method, name='forkmerge', **kwargs):
    from .forkmerge import ForkMerge
    stop_gradient_at = kwargs.get('stop_gradient_at', [])
    fm = ForkMerge(name=name, merge_layer=merge_method,
                   stop_gradient_at=stop_gradient_at)
    self.add(fm)
    return fm

  def add(self,
          f=None,
          inter_type=pedia.cascade,
          name=None,
          return_net=False,
          as_output=False,
          output_name=None,
          loss_identifier=None,
          target_key=None,
          loss_coef=1.0):
    """Add a net or a layer in to this model
    :param f: \in (Net, Layer)
    :param inter_type: inter-connection type
    :return: f or f's container
    """
    # If add an empty net
    if f is None:
      # TODO: inter_type add/concat/prod is deprecated
      #       use add_forkmerge instead
      name = self._get_new_name(inter_type) if name is None else name
      net = Net(name, level=self._level + 1, inter_type=inter_type)
      self.children.append(net)
      return net
    # Forbid adding nets of type other than 'cascade' using this method
    assert inter_type == pedia.cascade

    # If add a function to this net
    container = self
    if isinstance(f, Input):
      # If f is a placeholder
      self.input_ = f
    elif (isinstance(f, Net) or not self.is_root or
          self._inter_type not in (self.CASCADE, self.RECURRENT)):
      # Net should be added directly into self.children of any net
      # Layer should be added directly into self.children for non-cascade nets
      container = self._safe_add(f)
    elif isinstance(f, Layer):
      # If layer is a nucleus or the 1st layer added into this Net
      if f.is_nucleus or len(self.children) == 0:
        self._add_new_subnet(f)
      # Otherwise, add this layer to last Net of self.children
      container = self.add_to_last_net(f, only_cascade=True)
    else: raise ValueError('!! Object added to a Net must be a Layer or a Net')

    # Register output slot if necessary
    def _handle_error_injection():
      """This is a compromise"""
      if not as_output: return
      assert self.is_root
      self._output_slots.append(OutputSlot(
        self, f, loss=loss_identifier, loss_coef=loss_coef, name=output_name,
        target_key=target_key, last_only=False))
    _handle_error_injection()

    if return_net: return container
    else: return f

  # endregion : Public Methods


  # region : Private Methods

  def _safe_add(self, f):
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
    f = (context.customized_outer_loss_f_net if outer
         else context.customized_loss_f_net)
    if callable(f):
      loss_list = f(self)
      assert isinstance(loss_list, list)
      return loss_list
    else: return []

  def _get_extra_loss(self):
    loss_tensor_list = context.loss_tensor_list
    assert isinstance(loss_tensor_list, list)

    # (1) Add customized losses
    customized_loss = self._get_customized_loss()
    if customized_loss:
      loss_tensor_list += customized_loss

    # (2) Add regularized losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_tensor_list.extend(reg_losses)
    # (2-A) Add global l2 loss
    if hub.global_l2_penalty > 0:
      loss_tensor_list.append(hub.global_l2_penalty * tf.add_n(
        [tf.nn.l2_loss(v) for v in self.decayable_vars]))

    # Add-up all extra losses
    if loss_tensor_list:
      result = tf.add_n(loss_tensor_list, 'extra_loss')
    else: result = None

    # Show loss list (usually for debugging)
    if hub.show_extra_loss_info and loss_tensor_list:
      console.show_info('Extra losses:')
      for loss_tensor in loss_tensor_list:
        assert isinstance(loss_tensor, tf.Tensor)
        console.supplement(loss_tensor.name, level=2)
      console.split()
    return result

  def _gen_injection_loss(self):
    loss_tensors = []
    for slot in self.output_slots:
      assert isinstance(slot, OutputSlot)
      # Do auto plug
      loss_tensor = slot.auto_plug()
      if loss_tensor is not None: loss_tensors.append(loss_tensor)
    if not loss_tensors: return None
    loss_tensor_sum = tf.add_n(loss_tensors, name='injection_loss')
    console.show_status('{} loss injected'.format(len(loss_tensors)))
    return loss_tensor_sum

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
    """Extract variables to export"""
    get_key = lambda v: '/'.join(v.name.split('/')[1:])
    def add_to_dict(v): context.variables_to_export[get_key(v)] = v

    if hub.export_weights:
      for v in self.weight_vars: add_to_dict(v)

    # if hub.export_masked_weights and hub.pruning_rate_fc > 0:
    if hub.export_masked_weights:
      from tframe.advanced.prune.pruner import Pruner
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

    # modified on 2022-09-02
    monitor.register_weights(self.var_list)
    # weights of type tf.Variable
    # monitor.register_weights(self.weight_vars)

    # TODO: register masked_weights and sparse weights

  # endregion : Build-in extractors

