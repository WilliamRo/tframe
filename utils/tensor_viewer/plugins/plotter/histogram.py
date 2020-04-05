import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def histogram(
    subplot, values, val_range=None, title='Distribution', y_lim_pct=0.5):

  assert isinstance(subplot, plt.Axes) and isinstance(values, np.ndarray)
  # values for 1-D distribution must be flattened
  if len(values.shape) > 1: values = values.flatten()

  # Plot 1-D histogram
  subplot.hist(values, bins=50, facecolor='#cccccc', range=val_range)
  subplot.set_title(title)
  subplot.set_xlabel('Magnitude')
  subplot.set_ylabel('Density')

  # ~
  def to_percent(y, _):
    usetex = matplotlib.rcParams['text.usetex']
    pct = y * 100.0 / values.size
    return '{:.1f}{}'.format(pct, r'$\%$' if usetex else '%')
  subplot.yaxis.set_major_formatter(FuncFormatter(to_percent))

  subplot.set_aspect('auto')
  subplot.grid(True)

  subplot.set_ylim([0.0, y_lim_pct * values.size])




