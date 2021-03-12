from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.core.function import Function
from tframe.layers.layer import Layer
from tframe.layers.common import Linear
from tframe.layers.convolutional import Conv2D
from tframe.layers.merge import Merge
from tframe.nets.net import Net


class ForkMerge(Net):
  """A sub-network has the following topology:

       fork           merge
         _______________
       /                \
   -> x ---------------- y ->
      \                /
      ...           ...
       \____________/

  """

  def __init__(self, merge_layer, name='forkmerge', **kwargs):
    # Call parent's initializer
    super(ForkMerge, self).__init__(name, **kwargs)
    # Specialized fields
    if isinstance(merge_layer, Layer): self.merge_layer = merge_layer
    else: self.merge_layer = Merge(merge_layer, **kwargs)
    self._branch_dict = {}
    self._stop_gradient_keys = kwargs.get('stop_gradient_at', [])
    self._kwargs = kwargs


  @property
  def structure_detail(self):
    rows, total_params, dense_total = [], 0, 0
    for child in self.children:
      assert isinstance(child, Net)
      # Set output ID for each child
      child.children[-1].set_output_id()
      # Get details
      _rows, num, dense_num = child.structure_detail
      # Add indentation to each layer
      for i, row in enumerate(_rows):
        prefix = ' ' * 3 if i > 0 else '=> '
        # Add indentation according to level
        prefix = ' ' * 3 * (self._level - 1) + prefix
        # Prefix row
        _rows[i][0] = prefix + row[0]
      # Accumulate stuff
      rows += _rows
      total_params += num
      dense_total += dense_num

    # Consider merge layer
    row, num, dense_num = self._get_layer_detail(self.merge_layer)
    row[0] += '({})'.format(','.join(
      [child.children[-1].output_id_str for child in self.children]))
    rows.append(row)
    total_params += num
    dense_total += dense_num
    # Return
    return rows, total_params, dense_total


  def structure_string(self, detail=True, scale=True):
    return '{}({})'.format(self.name, ', '.join(
      [child.structure_string(detail, scale) for child in self.children] +
      [self.merge_layer.get_layer_string(False, False)]))


  def add_to_branch(self, branch_key, layer):
    # Get the Net to add layer
    if branch_key not in self._branch_dict:
      self._branch_dict[branch_key] = self.add(name=str(branch_key))
    branch = self._branch_dict[branch_key]
    # Add layer to branch
    assert isinstance(layer, Layer) and isinstance(branch, Net)
    return branch.add(layer)


  def _link(self, x:tf.Tensor, **kwargs):
    x_stop = tf.stop_gradient(x)
    output_list = [net(x_stop) if key in self._stop_gradient_keys else net(x)
                   for key, net in self._branch_dict.items()]
    return self.merge_layer(output_list)


class ForkMergeDAG(Net):
  """A more general fork-merge neural module that allows the structure between
     input and output operation to be any DAG, similar to what has been used
     in NAS-X01 serial papers. """

  def __init__(self, vertices, edges, name='DAG', transform=None,
               transform_kwargs=None, **kwargs):
    """A neural network module with one input and one output. The internal
    structure is represented as a DAG. Concatenation is used for merging
    inputs for multiple predecessors. One exception is the shortcut between
    module input and output, where the module input will be added to the
    merged tensor from other internal branches after necessary transformation.

    SYNTAX:
    -------
      model = Classifier('GooNet')
      model.add(Input(shape=...))

      fm_dag = ForkMergeDAG(
        [Conv2D(3x3), MaxPool2D(3x3), Conv2D(3x3), Conv2D(1x1), Conv2D(3x3)],
        edges='100111;10000;1000;100;10;1', name='DAG Example')

    :param vertices: list or tuple, each entry can be a Function or
                     list/tuple of Functions
    :param edges: a string or matrix (upper triangular) representing graph edge
    :param name: network name
    :param transform: shortcut transformation, if not provided, default
                      layer for transformation will be used
    :param transform_kwargs: keyword arguments for shortcut transformation
    :param kwargs: other keyword arguments
    """
    # Call parent's initializer
    super(ForkMergeDAG, self).__init__(name, **kwargs)
    # Attributes
    self.vertices = vertices
    self.edges = edges
    self.adj_mat = None
    self._init_graph()
    # Transformation
    if transform is not None: assert isinstance(transform, Function)
    self.transform = transform
    if transform_kwargs is None: transform_kwargs = {}
    self.transform_kwargs = transform_kwargs

    # Buffers
    self._predecessor_dict = {}
    self._front_vertices = []
    self._merged_dict = {}


  @property
  def structure_detail(self):
    rows, total_params, dense_total = [], 0, 0

    # Set id for predecessors
    for pred_list in self._predecessor_dict.values():
      for layer in pred_list: layer.set_output_id()

    for i, func in enumerate(self.children):
      assert isinstance(func, (Layer, Net))
      if isinstance(func, Layer):
        row, num, dense_num = self._get_layer_detail(func)
        _rows = [row]
      elif isinstance(func, Net):
        assert False  # Ban this option for now
        _rows, num, dense_num = func.structure_detail
      else: raise TypeError('!! Unknown function type `{}`'.format(type(func)))

      # Indent or add <= to front functions
      prefix = '' + ' ' * 3 * (self._level - 1)
      if i != len(self.children) - 1:
        prefix += '=> ' if func in self._front_vertices else ' ' * 3

      # Add suffix if necessary
      if func in self._predecessor_dict: prefix += '{}|'.format(''.join(
        [f.output_id_str for f in self._predecessor_dict[func]]))

      # Modify rows
      _rows[0][0] = prefix + _rows[0][0]

      # Accumulate stuff
      rows += _rows
      total_params += num
      dense_total += dense_num

    # Return
    return rows, total_params, dense_total


  def _init_graph(self):
    # Check vertices first
    assert isinstance(self.vertices, (list, tuple))
    vertices = []
    for vertex in self.vertices:
      if not isinstance(vertex, (list, tuple)): vertex = [vertex]
      for f in vertex: assert isinstance(f, Function)
      vertices.append(vertex)
    # Set vertices back
    self.vertices = vertices

    mat_size = len(self.vertices) + 2
    mat_shape = (mat_size, mat_size)
    # Check edges and formalize adjacent matrix
    if isinstance(self.edges, str):
      # Parse edge string
      rows = self.edges.split(';')
      # (1) check row number
      assert len(rows) == mat_size - 1
      self.adj_mat = np.zeros(shape=mat_shape, dtype=np.bool)
      for i, row in enumerate(rows):
        # (2) each row should represent an upper-triangular matrix row
        assert isinstance(row, str) and len(row) == mat_size - i - 1
        for j, c in enumerate(row):
          assert c in '01'
          self.adj_mat[i, j + i + 1] = c == '1'
    else:
      assert isinstance(self.edges, np.ndarray)
      self.adj_mat = self.edges.astype(dtype=np.bool)

    # Check adjacent matrix
    assert self.adj_mat.shape == mat_shape
    assert np.allclose(self.adj_mat, np.triu(self.adj_mat))
    if not all([
      all([np.sum(self.adj_mat[:, j]) > 0 for j in range(1, mat_size)]),  # in
      all([np.sum(self.adj_mat[i, :]) > 0 for i in range(mat_size - 1)]), # out
    ]): raise AssertionError('!! Adjacent matrix {} is illegal'.format(
      self.adj_mat))


  def _link(self, input_:tf.Tensor, **kwargs):
    # Output tensors of each vertex
    outputs = [input_]
    output_layers = {}
    for j, funcs in enumerate(self.vertices):
      # Add funcs to children list
      for f in funcs: self.add(f)

      # Get input tensors according to adjacent matrix
      #    in|vertices |out
      # j = 0|1 2 3 4 5|6          <= `outputs` indices
      #   [[+|1 0 0 1 1|1] 0       <= len(vertices) == 5
      #    [ |+ 1 0 0 0|0] 1
      #    [ |  + 1 0 0|0] 2       <= adjacent matrix
      #    [ |    + 0 1|0] 3
      #    [ |      + 1|0] 4
      #    [ |        +|1] 5
      #    [ |         |+]]
      input_tensors = [
        t for i, t in enumerate(outputs) if self.adj_mat[i, j + 1]]
      # Take down predecessors for structure details
      predecessors = [fs[-1] for i, fs in enumerate(self.vertices)
                      if self.adj_mat[i + 1, j + 1]]
      if predecessors: self._predecessor_dict[funcs[0]] = predecessors
      # Add input
      if self.adj_mat[0, j + 1]: self._front_vertices.append(funcs[0])
      # Merge tensor
      x, _ = self._internal_merge(input_tensors)
      # Feed merged tensor to function
      for f in funcs: x = f(x)
      outputs.append(x)
      output_layers[x] = f

    # Final merge
    input_tensors = [tensor for i, tensor in enumerate(outputs)
                     if self.adj_mat[i, -1] and i != 0]
    y, internal_final_layer = self._internal_merge(input_tensors)
    if internal_final_layer is not None:
      self._predecessor_dict[internal_final_layer] = [
        fs[-1] for i, fs in enumerate(self.vertices) if self.adj_mat[i + 1, -1]]
      self.add(internal_final_layer)
    else:
      assert len(input_tensors) == 1
      internal_final_layer = output_layers[input_tensors[0]]

    # Clear predecessor_dict for conciser structure detail
    for i, f in enumerate(self.children):
      if f not in self._predecessor_dict: continue
      preds = self._predecessor_dict[f]
      if len(preds) == 1 and preds[0] is self.children[i - 1]:
        self._predecessor_dict.pop(f)

    if not self.adj_mat[0, -1]: return y
    # Add shortcut if specified
    x = input_

    # Transform if necessary
    transform_layer = None
    input_shape, output_shape = input_.shape.as_list(), y.shape.as_list()
    is_aligned = input_shape == output_shape
    if not is_aligned and self.transform is None:
      # Automatically transform according to input shape
      # But firstly make sure this is possible
      if not all([
        len(input_shape) == len(output_shape),
        len(input_shape) in (2, 4),
        all([dx == dy for dx, dy in zip(input_shape[:-1], output_shape[:-1])]),
      ]): raise AssertionError(
        '!! Can not auto-transform input shape {} to target shape {}'.format(
          input_shape, output_shape))

      if len(input_shape) == 2:
        # For dense layers
        transform_layer = Linear(output_dim=output_shape[-1])
      else:
        # For convolutional layers (channel last)
        transform_layer = Conv2D(output_shape[-1], kernel_size=1)
    elif self.transform is not None:
      transform_layer = self.transform(**self.transform_kwargs)

    if transform_layer is not None:
      self.add(transform_layer)
      x = transform_layer(x)
      # Add information for structure detail
      self._front_vertices.append(transform_layer)

    # Add shortcut
    sum_merge_layer = self.add(Merge(Merge.SUM))
    y = sum_merge_layer([x, y])
    self._predecessor_dict[sum_merge_layer] = [internal_final_layer]
    if transform_layer is not None:
      self._predecessor_dict[sum_merge_layer].append(transform_layer)
    return y


  def _internal_merge(self, inputs):
    assert isinstance(inputs, list) and len(inputs) > 0
    if len(inputs) == 1: return inputs[0], None
    # Roughly make sure all tensors in inputs can be concatenated
    assert all([len(x.shape) == len(inputs[0].shape) for x in inputs])
    # Use concatenation to merge
    key = tuple(inputs)
    # Check merged_dict to avoid duplicated merge operation
    if key in self._merged_dict: return self._merged_dict[key]
    merge_layer = Merge(Merge.CONCAT)
    merged_tensor = merge_layer(inputs)
    self._merged_dict[key] = merged_tensor
    return merged_tensor, merge_layer



















