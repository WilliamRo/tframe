import re
from collections import OrderedDict

import numpy as np

from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView
from .plotter import histogram



def modifier(v_dict):
  assert isinstance(v_dict, OrderedDict)
  new_dict = OrderedDict()
  for key, values in v_dict.items():
    # if 'distribution' not in key: continue
    new_key = key
    new_dict[new_key] = VariableWithView(values, view=view)
    print('>> {} added to variable_dict'.format(new_key))
  v_dict.clear()
  v_dict.update(new_dict)


def view(self, param_list):
  from tframe.utils.tensor_viewer.variable_viewer import VariableViewer
  assert isinstance(param_list, list) and isinstance(self, VariableViewer)
  # Get range
  w_range = [np.min(param_list), np.max(param_list)]
  # Get weights
  theta = param_list[self.index]
  assert isinstance(theta, np.ndarray)
  theta = theta.flatten()

  # Hide ax2
  self.set_ax2_invisible()

  # Plot histogram
  title = 'Distribution ({} total)'.format(theta.size)
  histogram.histogram(self.subplot, theta, val_range=w_range, title=title)


plugin = Plugin(dict_modifier=modifier)
