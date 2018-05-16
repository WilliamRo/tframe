def check(inputs, types):
  if not isinstance(inputs, (tuple, list)): inputs = (inputs,)
  if not isinstance(types, (tuple, list)): types = (types,)
  assert len(inputs) == len(types)
  for obj, type_ in zip(inputs, types):
    if not isinstance(obj, type_):
      raise TypeError('!! Object {} must be an instance of {}'.format(
        obj, type_))
  if len(inputs) == 1: return inputs[0]
  else: return inputs

