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
  if not isinstance(inputs, tuple):
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
  if len(inputs) == 1: return inputs[0]
  else: return inputs

