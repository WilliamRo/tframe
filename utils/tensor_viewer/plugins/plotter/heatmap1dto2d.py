import numpy as np

import matplotlib.pyplot as plt


def heatmap2d(subplot, array, title=None, folds=5, v_range=None,
              min_color=(1., 1., 1.), max_color=(1., 0., 0.), grey=0.7):
  """
  :param array: numpy array
  :param folds: height of the image to plot
  :param v_range: value range, a tuple/list of 2 float number. None by default
  :param min_color: color of pixel with min value, a tuple/list of 3 float
                     numbers between 0. and 1.
  :param max_color: color of pixel with max value, a tuple/list of 3 float
                     number between 0. and 1.
  """
  # Check subplot and array
  assert isinstance(subplot, plt.Axes) and isinstance(array, np.ndarray)
  assert isinstance(folds, int) and folds > 0
  # Check v_range
  v_min, v_max = v_range if v_range else (min(array), max(array))
  assert v_max - v_min > 0.

  # Create a grey line
  size = array.size
  width = int(np.ceil(size / folds))
  max_color, min_color = [
    np.reshape(v, newshape=[1, 3]) for v in (max_color, min_color)]
  line = np.ones(shape=[width * folds, 3]) * grey

  # Map array into pixels with color and put them into the line
  values = np.maximum(0, array - v_min) / (v_max - v_min)
  values = np.stack([values] * 3, axis=1)
  values = values * (max_color - min_color) + min_color
  line[:size] = values

  # Fold line to image
  img = np.reshape(line, newshape=(folds, width, 3))

  # Plot image
  subplot.imshow(img, interpolation='none')

  # Hide y axis
  subplot.yaxis.set_ticks([])

  # Set grid off
  subplot.axis('off')

  # Set title if provided
  if isinstance(title, str): subplot.set_title(title)




