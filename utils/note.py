from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
from collections import OrderedDict


class Note(object):
  """A Note is held be an agent"""
  def __init__(self):
    self._lines = []

    # All lists below must have the same length, these are for TENSOR VIEWER
    self._steps = []
    self._scalars = OrderedDict()
    self._tensors = OrderedDict()

    # Configurations and criteria for SUMMARY VIEWER
    self._configs = OrderedDict()
    self._criteria = OrderedDict()

  # region : Properties

  # region : For TensorViewer

  @property
  def step_array(self):
    return np.array(self._steps) / 1000

  @property
  def scalar_dict(self):
    sd = OrderedDict()
    for k, v in self._scalars.items():
      sd[k] = np.array(v)
    return sd

  @property
  def tensor_dict(self):
    td = OrderedDict()
    for k, v in self._tensors.items():
      td[k] = v
    return td

  # endregion : For TensorViewer

  # region : For SummaryViewer

  @property
  def contain_tensors(self):
    if getattr(self, '_tensors', None) is None: return False
    return len(self._tensors) > 0

  @property
  def configs(self):
    assert isinstance(self._configs, dict)
    return self._configs

  @property
  def criteria(self):
    assert isinstance(self._criteria, dict)
    return self._criteria

  # endregion : For SummaryViewer

  @property
  def content(self):
    return '\n'.join(self._lines)

  # endregion : Properties

  # region : Public Methods

  # region : For TensorViewer

  def take_down_scalars_and_tensors(self, step, scalars, tensors):
    assert isinstance(scalars, dict) and isinstance(tensors, dict)
    # Take down step
    self._steps.append(step)
    # Take down scalars
    self._append_to_dict(self._scalars, scalars)
    # Take down parameters
    self._append_to_dict(self._tensors, tensors)

  # endregion : For TensorViewer

  # region : For SummaryViewer

  def put_down_configs(self, config_dict):
    assert isinstance(config_dict, dict)
    self._configs = config_dict

  def put_down_criterion(self, name, value):
    assert isinstance(name, str) and np.isscalar(value)
    self._criteria[name] = value

  # endregion : For SummaryViewer

  def write_line(self, line):
    assert isinstance(line, str)
    self._lines.append(line)

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
    for dict_ in [self._scalars, self._tensors]:
      for lst in dict_.values():
        assert isinstance(lst, list) and len(lst) == l

  # endregion : Private Methods
