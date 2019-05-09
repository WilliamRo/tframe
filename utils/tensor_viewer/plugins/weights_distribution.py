import re
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from matplotlib.ticker import FuncFormatter

from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView


def modifier(v_dict):
  assert isinstance(v_dict, OrderedDict)
  new_dict = OrderedDict()
  for key, values in v_dict.items():
    if not 'weights' in key: return
    new_key = key + '_hist'
    values = [v[v>0] for v in np.abs(values)]
    new_dict[new_key] = VariableWithView(values, view=view)
    print('>>  {} added to variable_dict'.format(new_key))
  v_dict.update(new_dict)


def view(self, weights_list):
  from tframe.utils.tensor_viewer.variable_viewer import VariableViewer
  assert isinstance(weights_list, list) and isinstance(self, VariableViewer)
  # Get weights
  weights = weights_list[self.index]
  assert isinstance(weights, np.ndarray)
  # Plot
  self.set_ax2_invisible()

  self.subplot.hist(weights, bins=50, facecolor='#cccccc')
  self.subplot.set_title(
    'Weights magnitude distribution ({} total)'.format(weights.size))
  self.subplot.set_xlabel('Magnitude')
  self.subplot.set_ylabel('Density')

  def to_percent(y, _):
    usetex = matplotlib.rcParams['text.usetex']
    pct = y * 100.0 / weights.size
    return '{:.1f}{}'.format(pct, r'$\%$' if usetex else '%')
  self.subplot.yaxis.set_major_formatter(FuncFormatter(to_percent))

  self.subplot.set_aspect('auto')
  self.subplot.grid(True)

  # self.subplot.invert_yaxis()

plugin = Plugin(dict_modifier=modifier)
