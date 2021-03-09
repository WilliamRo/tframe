import numpy as np


class HyperParameter(object):

  token = 'HP'

  def __init__(self, name):
    self.name = name
    self.hp_type = None
    self.scale = None

  @property
  def option_str(self):
    return '({}) {}'.format(self.token, self._option_str)

  @property
  def _option_str(self):
    return 'Unknown'

  def within(self, value):
    raise NotImplementedError

  def to_vector_list(self, val):
    raise NotImplementedError

  def vector_to_hp(self, vector):
    raise NotImplementedError



class FloatHP(HyperParameter):

  token = 'F'

  def __init__(self, name, v_min, v_max, scale=None):
    # Call parent's constructor
    super().__init__(name)
    # Specific variables
    legal_scales = ('uniform', 'log', 'log-uniform', None)
    assert scale in legal_scales
    if scale == legal_scales[1]: scale = legal_scales[2]
    self.scale = scale
    assert v_min < v_max
    self.v_min, self.v_max = v_min, v_max

  @property
  def _option_str(self):
    return '[{}, {}], {}'.format(self.v_min, self.v_max, self.scale)

  def within(self, value):
    return self.v_min <= value <= self.v_max

  def to_vector_list(self, val):
    assert self.within(val)
    return [val]

  def vector_to_hp(self, vector):
    assert isinstance(vector, float)
    return vector


class IntegerHP(FloatHP):

  token = 'I'

  def __init__(self, name, v_min, v_max, scale=None):
    assert all([isinstance(v, int) for v in (v_min, v_max)])
    # Call parent's constructor
    super().__init__(name, v_min, v_max, scale)

  def vector_to_hp(self, vector):
    assert isinstance(vector, (int, float))
    return int(np.round(vector))


class CategoricalHP(HyperParameter):

  token = 'C'

  def __init__(self, name, choices, hp_type=None, scale=None):
    # Call parent's constructor
    super().__init__(name)
    # Specific variables
    assert isinstance(choices, (list, tuple)) and len(choices) > 1
    self.choices = choices
    # If hp_type is set, this HP can be transformed to the corresponding type
    assert all([hp_type in (int, float, None, list, tuple),
                scale in ('uniform', 'log', 'log-uniform', None)])
    self.hp_type = hp_type
    self.scale = scale

  @property
  def _option_str(self):
    return '{}'.format(self.choices)

  def within(self, value):
    if self.hp_type in (int, float):
      return min(self.choices) <= value <= max(self.choices)
    return value in self.choices

  def _binary_to_vector_list(self, val):
    assert self.within(val) and len(self.choices) == 2
    return [0. if val == self.choices[0] else 1.]

  def to_vector_list(self, val):
    assert self.within(val)
    if len(self.choices) == 2: return self._binary_to_vector_list(val)
    vector_list = [1 if c == val else 0 for c in self.choices]
    assert sum(vector_list) == 1
    return vector_list

  def vector_to_hp(self, vector):
    if isinstance(vector, (int, float)):
      assert 0 <= vector <= 1
      return int(np.round(vector))
    return np.argmax(vector)

  def seek_myself(self):
    if self.hp_type in (None, list, tuple): return self
    scale = 'uniform' if self.scale is None else self.scale
    assert all([isinstance(c, self.hp_type) for c in self.choices])
    HPClass = {int: IntegerHP, float: FloatHP}[self.hp_type]
    return HPClass(self.name, min(self.choices), max(self.choices), scale)


class BooleanHP(CategoricalHP):

  token = 'B'

  def __init__(self, name):
    # Call parent's constructor
    super().__init__(name, choices=(True, False))



