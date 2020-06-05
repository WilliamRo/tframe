from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import colors


def get_rgb_by_name(name, dtype=float):
  rgb = np.array(colors.hex2color(colors.cnames[name]))
  if dtype is not float: rgb = (rgb * 255).astype(np.int)
  return rgb


def int2listedcmap(indices, cmap):
  assert isinstance(indices, np.ndarray) and isinstance(cmap, list)
  cmap = np.array([get_rgb_by_name(name) for name in cmap])
  return cmap[indices]


if __name__ == '__main__':
  from matplotlib import colors
  print(get_rgb_by_name('white', int))
  cmap = np.array([[-2, -4], [2, 4.0]])
  indices = np.array([[0, 1], [1, 0]])
  im = cmap[indices]
  print(indices)
  print(im)

