import re
from collections import OrderedDict

import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter

from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView


def _recursive_modify(v_dict):
  if len(v_dict) == 0: return
  assert isinstance(v_dict, dict)
  if isinstance(list(v_dict.values())[0], dict):
    for e_dict in v_dict.values(): _recursive_modify(e_dict)
    return
  # Here the values in v_dict must be lists



def modifier(v_dict):
  _recursive_modify(v_dict)
  return

  assert isinstance(v_dict, OrderedDict)
  new_dict = OrderedDict()
  for key, values in v_dict.items():
    s = re.search(r'(?<=weights_)\d+', key)
    if s is None: continue
    new_key = key + '_hist'
    # values = [v[v>0] for v in np.abs(values)]
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
  self.subplot.set_xlim([0, np.max(weights_list)])

  def to_percent(y, _):
    usetex = matplotlib.rcParams['text.usetex']
    pct = y * 100.0 / weights.size
    return '{:.1f}{}'.format(pct, r'$\%$' if usetex else '%')
  self.subplot.yaxis.set_major_formatter(FuncFormatter(to_percent))

  self.subplot.set_aspect('auto')
  self.subplot.grid(True)

  # y_lim = self.subplot.get_ylim()
  # self.subplot.set_ylim(y_lim[::-1])
  self.subplot.set_ylim((0.0, 8000.0))

plugin = Plugin(dict_modifier=modifier)
