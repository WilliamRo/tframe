from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import console, checker, pedia
from tframe.layers.convolutional import Conv2D
from tframe.layers.layer import Layer, single_input, Function
from tframe.utils import get_scale

from typing import Optional


class ShortCut(Layer):
  full_name = 'shortcut'
  abbreviation = 'shortcut'

  is_nucleus = False

  class Mode:
    SUM = 'sum'
    CONCATE = 'concate'

  @property
  def structure_tail(self):
    return '({})'.format(','.join(self.transformation_str_list))

  @property
  def transformation_str_list(self):
    result = []
    if self._transforms is None:
      # return [f.output_id_str for f in self.definitions]
      return [f.get_layer_string(True) if isinstance(f, Layer)
              else f.structure_string(True) for f in self.definitions]
    for f, t_list in zip(self.definitions, self._transforms):
      result.append(
        '->'.join([f.output_id_str] + [t.group_name for t in t_list]))
    return result

  def __init__(self, *definitions, axis=-1, mode='concate', transforms=None):
    if len(definitions) == 0:
      console.warning_with_pause('Nothing to be merged.')
    self.definitions = checker.check_type(definitions, Function)
    # for f in self.definitions: f.set_output_id()
    self.axis = axis
    # Check mode
    assert mode in (self.Mode.SUM, self.Mode.CONCATE)
    self.mode = mode
    self.full_name = mode
    self.abbreviation = mode
    # Check transforms
    if transforms is not None:
      assert isinstance(transforms, list)
      assert len(transforms) == len(self.definitions)
      for t in transforms: checker.check_type(t, Function)
    self._transforms = transforms

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    if len(self.definitions) == 0: return input_
    # Get tensor list to merge
    tensors = [input_]
    # Execute definitions if not executed
    for f in self.definitions:
      if f.output_tensor is None: _ = f(input_)
    if self._transforms is None:
      tensors += [f.output_tensor for f in self.definitions]
    else:
      for i, (f, t_list) in enumerate(zip(self.definitions, self._transforms)):
        y = f.output_tensor
        for j, t in enumerate(t_list):
          with tf.variable_scope('{}-{}'.format(i, j)): y = t(y)
        tensors.append(y)
    # Merge
    if self.mode == self.Mode.CONCATE: return tf.concat(tensors, axis=self.axis)
    elif self.mode == self.Mode.SUM: return tf.add_n(tensors)
    else: raise NotImplementedError

  def add_transformation(self, f, branch_id=0):
    assert isinstance(f, Function)
    if getattr(f, 'is_nucleus', False): self.is_nucleus = True
    # Initialize self._transform if necessary
    if self._transforms is None:
      self._transforms = [[] for _ in self.definitions]
    # Add f into transformation list
    self._transforms[branch_id].append(f)


class Merge(Layer):

  PROD = pedia.prod
  SUM = pedia.sum
  CONCAT = pedia.concat
  CROSS_CONCAT = 'cross-concat'
  CONCAT_SUM = 'concat-sum'
  HIGHWAY = 'highway'

  def __init__(self, merge_method, **kwargs):
    """This layer class provides some build-in merge method, including
    those listed in the class variables with capitalized names. When using
    `CONCAT_SUM` method, one needs to specify `sum_indices`. Other tensors
    will be concatenated first and added to those within `sum_indices`.
    Currently a more general design is not needed."""
    self.full_name, self.abbreviation = merge_method, merge_method
    self.merge_method = merge_method
    # Attributes for CONCAT method
    self._axis = kwargs.get('axis', -1)
    # Attributes for `CONCAT-SUM` method
    self._sum_indices = kwargs.get('sum_indices', (0,))
    if isinstance(self._sum_indices, int):
      self._sum_indices = (self._sum_indices,)
    if merge_method == self.CONCAT_SUM:
      self.full_name += ('-{}'.format(
        ','.join(self._sum_indices) if len(self._sum_indices) > 1 else 0))

    self.max_trim = kwargs.get('max_trim', 0)
    # Store other keyword arguments
    self.kwargs = kwargs

  def _link(self, *input_list, **kwargs):
    # Check input_list
    assert len(input_list) > 0
    if len(input_list) == 1: input_list = input_list[0]
    if not (isinstance(input_list, (list, tuple)) and len(input_list) > 1):
      raise ValueError('!! Illegal input tensors flow into merge layer.')

    # Slice if necessary
    input_list = self._check_input_list(input_list)

    # Merge according to specification
    if self.merge_method == self.SUM: return tf.add_n(input_list)
    elif self.merge_method == self.CONCAT:
      return tf.concat(input_list, axis=self._axis)
    elif self.merge_method == self.CROSS_CONCAT:
      assert len(input_list) == 2
      x: tf.Tensor = input_list[0]
      y: tf.Tensor = input_list[1]
      assert x.shape.as_list() == y.shape.as_list()
      xy = tf.multiply(x, y, name='cross')
      return tf.concat([x, y, xy], axis=self._axis)
    elif self.merge_method == self.PROD:
      output = input_list.pop()
      for tensor in input_list: output *= tensor
      return output
    elif self.merge_method == self.CONCAT_SUM:
      assert len(input_list) > 2
      assert 0 < len(self._sum_indices) <= len(input_list) - 2
      y = tf.concat([x for i, x in enumerate(input_list)
                     if i not in self._sum_indices], axis=self._axis)
      inputs = [x for i, x in enumerate(input_list) if i in self._sum_indices]
      inputs.append(y)
      return tf.add_n(inputs)
    elif self.merge_method == self.HIGHWAY:
      assert len(input_list) == 3
      x, x_bar, gate = input_list
      y = tf.multiply(gate, x) + tf.multiply(1. - gate, x_bar)
      return y
    else: raise KeyError('!! Unknown merge method {}'.format(self.merge_method))

  def _check_input_list(self, input_list):
    # Make sure each input has the same shape length
    shapes = [x.shape.as_list() for x in input_list]
    if not all([len(shape) == len(shapes[0]) for shape in shapes]):
      raise AssertionError('!! tensors to be merged must have a same rank')

    # TODO: some more checks should be done
    if self.merge_method not in (self.SUM, self.PROD): return input_list
    dims = [shape[self._axis] for shape in shapes]
    min_dim = min(dims)
    deltas = [d - min_dim for d in dims]
    if not any(deltas): return input_list

    # Try to automatically truncate overlong tensors
    if max(deltas) > self.max_trim: raise ValueError(
      '!! Failed to merge tensors because of unequal dimension of the'
      ' corresponding axis which can not be truncated automatically.')

    # Truncate overlong tensors
    begin, size = [[i] * len(shapes[0]) for i in (0, 1)]
    size[self._axis] = min_dim
    for i, delta in enumerate(deltas):
      if delta == 0: continue
      input_list[i] =tf.slice(input_list[i], begin, size)

    self.full_name += '[t]'
    return input_list

  @classmethod
  def Sum(cls, **kwargs):
    return Merge(cls.SUM, **kwargs)

  @classmethod
  def Prod(cls, **kwargs):
    return Merge(cls.PROD, **kwargs)

  @classmethod
  def Concat(cls, axis=-1, **kwargs):
    return Merge(cls.CONCAT, axis=axis, **kwargs)

  @classmethod
  def CrossConcat(cls, axis=-1, **kwargs):
    return Merge(cls.CROSS_CONCAT, axis=axis, **kwargs)

  @classmethod
  def ConcatSum(cls, sum_indices=(0,), **kwargs):
    return Merge(cls.CONCAT_SUM, sum_indices=sum_indices, **kwargs)

  @classmethod
  def Highway(cls, **kwargs):
    return Merge(cls.HIGHWAY, **kwargs)


class ConcatenateForGAN(Layer):
  full_name = 'concatenate'
  abbreviation = 'concat'

  def __init__(self, companions=None):
    """
    Initiate a concatenate layer
    :param companions: a dictionary with format:
                       {tensor0: insert_position0, ..., 
                        tensorN: insert_positionN}
                        companion tensors will be inserted into input list
                        at the specific position
    """
    # Check companion
    if companions is not None:
      for key in companions.keys():
        if not isinstance(key, tf.Tensor):
          raise TypeError('key must be a tensor')
        if not isinstance(companions[key], int):
          raise TypeError('value must be an integer')

    self._companions = companions

  def _link(self, inputs, **kwargs):
    if isinstance(inputs, tf.Tensor):
      inputs = [inputs]

    # Avoid that insert operation below changes the original list
    inputs = inputs.copy()

    assert isinstance(inputs, list)
    if not self._companions is None:
      for tensor in self._companions.keys():
        assert isinstance(tensor, tf.Tensor)
        inputs.insert(self._companions[tensor], tensor)

    # Check inputs
    if len(inputs) < 2:
      raise ValueError('inputs to concatenate layer must have a length'
                        ' larger than 1')

    # Prepare inputs for concatenation
    assert isinstance(inputs[0], tf.Tensor)
    leader_shape_tensor = tf.shape(inputs[0])
    leader_shape = inputs[0].get_shape().as_list()

    for i in range(1, len(inputs)):
      assert isinstance(inputs[i], tf.Tensor)
      shape = inputs[i].get_shape().as_list()
      shape_tensor = tf.shape(inputs[i])
      ones = tf.ones(
        [leader_shape_tensor[0]] + leader_shape[1:-1] + [shape[-1]])

      target_shape = ([shape_tensor[0]] + [1]*(len(leader_shape) - 2)
                      + [shape[-1]])
      reshaped = tf.reshape(inputs[i], target_shape)

      inputs[i] = reshaped * ones

    result = tf.concat(inputs, axis=len(leader_shape) - 1)
    self.neuron_scale = get_scale(result)
    return result


class Bridge(Layer):

  full_name = 'bridge'
  abbreviation = 'bridge'
  is_nucleus = False

  def __init__(self, conv_layer: Conv2D,
               guest_is_larger: Optional[bool] = None,
               guest_first: bool = True):
    """Concatenate 2 convolutional mappings with necessary cropping"""
    self.conv_layer = conv_layer
    self.guest_is_larger = guest_is_larger
    self.guest_first = guest_first

  @property
  def structure_tail(self):
    return '(*,{})'.format(self.conv_layer.output_id_str)

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    # Get guest
    a = self.conv_layer.output_tensor
    assert isinstance(a, tf.Tensor)
    # Define utilities
    crop = lambda src, tgt: src[:, :tgt.shape[1], :tgt.shape[2]]

    # Crop if necessary
    if self.guest_is_larger is True: a = crop(a, x)
    elif self.guest_is_larger is False: x = crop(x, a)

    # Concatenate and return
    tensors = [a, x] if self.guest_first else [x, a]
    return tf.concat(tensors, axis=-1)



