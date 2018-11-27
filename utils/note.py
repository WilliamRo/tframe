from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np


class Note(object):
  """A Note is held be an agent"""
  def __init__(self):
    self._lines = []
    # TODO: BETA
    # Configurations and criteria
    self._configs = {}
    self._criteria = {}
    # All lists below (may be as values of some dict) must have the same length
    self._steps = []
    self._scalars = {}
    self._parameters = {}

  # region : Properties

  @property
  def configs(self):
    assert isinstance(self._configs, dict)
    return self._configs

  @property
  def criteria(self):
    assert isinstance(self._criteria, dict)
    return self._criteria

  @property
  def content(self):
    return '\n'.join(self._lines)

  @property
  def loss_array(self):
    key = 'Loss'
    assert key in self._scalars.keys()
    return np.array(self._scalars[key])

  @property
  def step_array(self):
    return np.array(self._steps) / 1000

  @property
  def variable_dict(self):
    return {k: v for k, v in self._parameters.items() if len(v[0].shape) > 1}

  # endregion : Properties

  # region : Public Methods

  def write_line(self, line):
    assert isinstance(line, str)
    self._lines.append(line)

  def take_down_params(self, step, scalars, params):
    assert isinstance(scalars, dict) and isinstance(params, dict)
    # Take down step
    self._steps.append(step)
    # Take down scalars
    self._append_to_dict(self._scalars, scalars)
    # Take down parameters
    self._append_to_dict(self._parameters, params)

  def put_down_criterion(self, name, value):
    assert isinstance(name, str) and np.isscalar(value)
    self._criteria[name] = value

  def save(self, file_name):
    self._check_before_dump()
    with open(file_name, 'wb') as f:
      pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

  @staticmethod
  def load(file_name):
    with open(file_name, 'rb') as f:
      return pickle.load(f)

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _append_to_dict(dst, src):
    assert isinstance(dst, dict) and isinstance(src, dict)
    for key, value in src.items():
      # If dst is empty, init the corresponding list
      if key not in dst.keys():
        dst[key] = []
      # Append value to dst[key]
      assert isinstance(dst[key], list)
      dst[key].append(value)

  def _check_before_dump(self):
    l = len(self._steps)
    for dict_ in [self._scalars, self._parameters]:
      for lst in dict_.values():
        assert isinstance(lst, list) and len(lst) == l

  # endregion : Private Methods
