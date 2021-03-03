import numpy as np


class HyperParameter(object):

  token = 'HP'

  def __init__(self, name):
    self.name = name

  @property
  def option_str(self):
    return '<{}> {}'.format(self.token, self._option_str)

  @property
  def _option_str(self):
    return 'Unknown'

  def within(self, value):
    raise NotImplementedError


class FloatHP(HyperParameter):

  token = 'F'

  def __init__(self, name, v_min, v_max, scale='uniform'):
    # Call parent's constructor
    super().__init__(name)
    # Specific variables
    assert scale in ('uniform', 'log')
    self.scale = scale
    assert v_min < v_max
    self.v_min, self.v_max = v_min, v_max

  @property
  def _option_str(self):
    return '[{}, {}], {}'.format(self.v_min, self.v_max, self.scale)

  def within(self, value):
    return self.v_min <= value <= self.v_max


class IntegerHP(FloatHP):

  token = 'I'

  def __init__(self, name, v_min, v_max, scale='uniform'):
    assert all([isinstance(v, int) for v in (v_min, v_max)])
    # Call parent's constructor
    super().__init__(name, v_min, v_max, scale)


class CategoricalHP(HyperParameter):

  token = 'C'

  def __init__(self, name, choices):
    # Call parent's constructor
    super().__init__(name)
    # Specific variables
    assert isinstance(choices, (list, tuple, set))
    self.choices = choices

  @property
  def _option_str(self):
    return '{}'.format(self.choices)

  def within(self, value):
    return value in self.choices


class BooleanHP(CategoricalHP):

  token = 'B'

  def __init__(self, name):
    # Call parent's constructor
    super().__init__(name, choices=(True, False))



