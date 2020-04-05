import re
from collections import OrderedDict

import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter

from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView
from .plotter import histogram


prefix = 'weights_'

def modifier(v_dict):
  assert isinstance(v_dict, OrderedDict)
  new_dict = OrderedDict()
  for key, values in v_dict.items():
    s = re.search(r'(?<=' + prefix + r')\d+', key)
    if s is None: continue
    new_key = key + '_hist'
    new_dict[new_key] = VariableWithView(values, view=view)
    print('>> {} added to variable_dict'.format(new_key))
  v_dict.update(new_dict)


def view(self, weights_list):
  from tframe.utils.tensor_viewer.variable_viewer import VariableViewer
  assert isinstance(weights_list, list) and isinstance(self, VariableViewer)
  # Get range
  w_range = [np.min(weights_list), np.max(weights_list)]
  # Get weights
  weights = weights_list[self.index]
  assert isinstance(weights, np.ndarray)
  weights = weights.flatten()

  # Hide ax2
  self.set_ax2_invisible()

  # Plot histogram
  title = 'Weights magnitude distribution ({} total)'.format(weights.size)
  histogram.histogram(self, weights, val_range=w_range, title=title)


plugin = Plugin(dict_modifier=modifier)
