from tframe import tf
import numpy as np

from tframe.utils import misc


def try_str2float(s):
  try: return float(s)
  except ValueError: return s


def check_conv_size(size, dim=2, dtype=tuple):
  if not isinstance(size, (tuple, list)): size = (size,) * dim
  for s in size: check_positive_integer(s, name='size component `{}`'.format(s))
  return dtype(size)


def check_tensor_shape(tensor1, tensor2, name1=None, name2=None):
  if name1 is None: name1 = misc.retrieve_name(tensor1)
  if name2 is None: name2 = misc.retrieve_name(tensor2)
  assert isinstance(name1, str) and isinstance(name2, str)
  assert isinstance(tensor1, tf.Tensor) and isinstance(tensor2, tf.Tensor)
  shape1, shape2 = tensor1.shape.as_list(), tensor2.shape.as_list()
  if shape1 != shape2:
    raise ValueError(
      '!! {}.shape({}) should be equal with {}.shape({}) '.format(
        name1, shape1, name2, shape2))


def check_gate(inputs):
  objects = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
  for obj in objects:
    if not np.isreal(obj):
      raise TypeError('!! {} must be a number.'.format(obj))
    if not 0 <= obj <= 1:
      raise AssertionError('!! {} should be in [0, 1]'.format(obj))
  return inputs


def check_fetchable(inputs):
  return check_type_v2(inputs, (tf.Tensor, tf.Variable, tf.Operation))


def check_type_v2(inputs, types):
  """
  SYNTAX:
  (1) check(value, (int, float))
  (2) check([v1, v2, ..., vN], (int, float))
  (2) check([v1, v2, ..., vN], tf.Tensor)
  """
  objects = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
  for obj in objects:
    if not isinstance(obj, types):
      raise TypeError('!! Object {} must be an instance of {}'.format(
        obj, types))
  return inputs


def check_scalar_list(inputs, raise_error=False):
  assert isinstance(inputs, list)
  for x in inputs:
    if not np.isscalar(x):
      if raise_error: raise TypeError('{} is not a scalar.'.format(x))
      return False
  return True


def check_type(inputs, type_tuples):
  """
  Check the types of inputs
  SYNTAX:
  (1) check(value, int)
  (2) check(value, (int, bool))
  (3) val1, val2 = check([val1, val2], tf.Tensor)
  (4) val1, val2 = check([val1, val2], (tf.Tensor, tf.Variable))

  :param inputs: \in {obj, tuple, list}
  :param type_tuples: \in {type, tuple of types, tuple of tuple of types}
  :return: a tuple of inputs
  """
  if isinstance(inputs, list): inputs = tuple(inputs)
  if not isinstance(type_tuples, (tuple, list)): type_tuples = (type_tuples,)
  inputs_are_tuple = isinstance(inputs, tuple)
  if not inputs_are_tuple:
    inputs = (inputs,)
    type_tuples = (type_tuples,)
  if len(inputs) > 1 and len(type_tuples) == 1:
    type_tuples = type_tuples * len(inputs)
  assert len(inputs) == len(type_tuples)
  for obj, type_tuple in zip(inputs, type_tuples):
    # Make sure type_tuple is a type or a tuple of types
    if not isinstance(type_tuple, tuple): type_tuple = (type_tuple,)
    for type_ in type_tuple: assert isinstance(type_, type)
    # Check obj
    if not isinstance(obj, type_tuple):
      raise TypeError('!! Object {} must be an instance of {}'.format(
        obj, type_tuple))
  # Return inputs
  if len(inputs) == 1 and not inputs_are_tuple: return inputs[0]
  else: return inputs


def check_positive_integer(x, allow_zero=False, name=None):
  if not isinstance(x, (int, np.integer)) or x < 0 or not allow_zero and x == 0:
    if name is None: name = misc.retrieve_name(x)
    raise ValueError('!! {} must be a positive integer'.format(name))
  return x


def get_range(rng):
  if not isinstance(rng, tuple) or len(rng) != 2:
    raise TypeError('!! Range must be a tuple of length 2')
  low, high = rng
  if not np.isscalar(low) or not np.isscalar(high):
    raise TypeError('!! Range should be a tuple consists of 2 scalars')
  if low >= high:
    raise AssertionError('!! Illegal range = ({}, {})'.format(low, high))
  return low, high


def check_callable(f, name=None, allow_none=True):
  if name is None: name = misc.retrieve_name(f)
  flag = True
  if not allow_none and f is None: flag = False
  if f is not None and not callable(f): flag = False
  if flag: return f
  else: raise TypeError('!! {} must be callable'.format(name))


def check(assertion, err_msg, err_type=AssertionError):
  if not assertion: raise err_type('!! {}'.format(err_msg))
