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
  # Here the values in v_dict must be lists, keys may include
  #  (weights_k, mask_k)_k pairs
  pairs = []
  for key in list(v_dict.keys()):
    scopes = re.findall(r'\w+/\w+/W\w*', key)
    if not scopes: continue
    assert len(scopes) == 1
    scope = scopes[0]
    # Find mask key
    mask_key = scope + '_mask'
    if not mask_key in v_dict.keys(): continue
    # Now pair is found
    pairs.append((scope, v_dict.pop(key), v_dict.pop(mask_key)))
  # Update v_dict
  for scope, weights, mask in pairs:
    assert isinstance(weights, list) and isinstance(mask, list)
    hist_key = scope + '_histogram'
    v_dict[scope + '_masked'] = [w * m for w, m in zip(weights, mask)]
    v_dict[hist_key] = VariableWithView((weights, mask), view=view)
    # Show status
    suffix = '..' * level if level > 0 else '>>'
    print(suffix + ' {} added to variable_dict'.format(hist_key))


def modifier(v_dict):
  print('>> Modifying by lottery ...')
  _recursive_modify(v_dict)


def view(self, pair):
  from tframe.utils.tensor_viewer.variable_viewer import VariableViewer
  assert isinstance(pair, tuple) and isinstance(self, VariableViewer)
  # Unwrap pair and get selected weights and mask
  weights_list, mask_list = pair
  weights, mask = weights_list[self.index], mask_list[self.index]
  assert isinstance(weights, np.ndarray) and isinstance(mask, np.ndarray)
  # Get flattened data
  masked_weights = weights[mask == 1]
  weights = weights.flatten()
  # Calculate fraction and range
  frac = 100.0 * masked_weights.size / weights.size
  w_range = ([np.min(weights_list), np.max(weights_list)]
             if self.unify_range else None)

  # - Plot
  # .. preparation
  self.set_ax2_invisible()
  # .. plot histogram
  bins = 50
  self.subplot.hist(weights, bins=bins, label='Dense',
                    facecolor='#cccccc', range=w_range)
  self.subplot.hist(masked_weights, bins=bins, label='Pruned',
                    facecolor='#ffaa00', range=w_range)
  self.subplot.legend(loc='best')

  # .. set title and label
  self.subplot.set_title(
    'Distribution, #{} ({:.2f}%) '.format(masked_weights.size, frac))
  self.subplot.set_xlabel('Value')
  self.subplot.set_ylabel('Density')
  # .. convert to density
  def to_percent(y, _):
    usetex = matplotlib.rcParams['text.usetex']
    pct = y * 100.0 / weights.size
    return '{:.1f}{}'.format(pct, r'$\%$' if usetex else '%')
  self.subplot.yaxis.set_major_formatter(FuncFormatter(to_percent))
  # .. post-process
  self.subplot.set_aspect('auto')
  self.subplot.grid(True)
  # .. prevent display upside down
  y_lim = self.subplot.get_ylim()
  if y_lim[0] > y_lim[1]: self.subplot.set_ylim(y_lim[::-1])

plugin = Plugin(dict_modifier=modifier)
