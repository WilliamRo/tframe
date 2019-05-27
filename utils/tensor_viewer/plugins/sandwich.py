import re

import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter

from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView


def _recursive_modify(v_dict, level=0):
  if len(v_dict) == 0: return
  assert isinstance(v_dict, dict)
  if isinstance(list(v_dict.values())[0], dict):
    for e_key, e_dict in v_dict.items():
      print('>> Modifying dict {} ...'.format(e_key))
      _recursive_modify(e_dict, level=level + 1)
    return
  # Here the values in v_dict must be lists
  for key in list(v_dict.keys()):
    if not re.fullmatch(r'dL/dS\[\d+\]', key): continue
    triangle_list = v_dict[key]
    new_list = []
    for triangle in triangle_list:
      assert isinstance(triangle, np.ndarray) and len(triangle.shape) == 2
      bottom = np.sum(triangle, axis=0, keepdims=True)
      new_list.append(np.concatenate(
        [triangle, np.zeros_like(bottom), bottom], axis=0))
    v_dict[key] = new_list


def modifier(v_dict):
  print('>> Modifying by sandwich ...')
  _recursive_modify(v_dict)


plugin = Plugin(dict_modifier=modifier)
