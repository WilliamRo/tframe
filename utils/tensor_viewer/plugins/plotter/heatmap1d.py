import numpy as np

import matplotlib.pyplot as plt


def linear_heatmap(
    subplot, array, title=None, horizontal=True, cmap='bwr', width=2,
    vmax=1, vmin=-1):
  assert isinstance(subplot, plt.Axes) and isinstance(array, np.ndarray)
  assert isinstance(width, int) and width >= 1

  # Stretch image
  img = np.stack([array.flatten()] * width, axis=0 if horizontal else 1)

  # Plot image
  subplot.imshow(img, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)

  # Hide y axis
  subplot.yaxis.set_ticks([])

  # Set grid off
  subplot.axis('off')

  # Set title if provided
  if isinstance(title, str): subplot.set_title(title)



