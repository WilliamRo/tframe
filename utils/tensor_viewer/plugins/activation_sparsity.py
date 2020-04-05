import re
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt

from tframe import checker
from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView
from tframe.utils.tensor_viewer.plugin import recursively_modify

from .plotter.histogram import histogram
from .plotter.heatmap1d import linear_heatmap
from .plotter.heatmap1dto2d import heatmap2d


def view(self, array_list):
  from tframe.utils.tensor_viewer.variable_viewer import VariableViewer
  assert isinstance(array_list, list) and isinstance(self, VariableViewer)

  # Handle things happens in VariableView.refresh method

  # Create subplots if not exists
  if not hasattr(self, 'sub211'):
    self.sub211 = self.figure.add_subplot(211, autoscale_on=True)
  if not hasattr(self, 'sub212'):
    self.sub212 = self.figure.add_subplot(212, autoscale_on=True)
  # Clear subplots
  self.sub211.cla()
  self.sub212.cla()
  # Hide subplot
  self.subplot.set_axis_off()

  # Hide ax2
  self.set_ax2_invisible()

  # Plot histogram

  # Get range
  a_range = [np.min(array_list), np.max(array_list)]
  # Get activation
  activation = array_list[self.index].flatten()
  title = 'Activation Distribution'
  histogram(self.sub211, activation, val_range=a_range, title=title)

  # Plot heat-map
  heatmap2d(self.sub212, activation, folds=5)

  # Tight layout
  self.figure.tight_layout()


def method(key, value):
  assert isinstance(key, str)
  if 'sog_gate' not in key: return value
  checker.check_type_v2(value, np.ndarray)
  # Make sure activation is 1-D array
  assert len(value[0].shape) == 1
  return VariableWithView(value, view)


def modifier(v_dict):
  assert isinstance(v_dict, OrderedDict)
  recursively_modify(method, v_dict, verbose=True)


plugin = Plugin(dict_modifier=modifier)
