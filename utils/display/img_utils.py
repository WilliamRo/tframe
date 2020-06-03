from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def int2listedcmap(indices, cmap):
  assert isinstance(indices, np.ndarray) and isinstance(cmap, list)
  from matplotlib import colors
  cmap = np.array([colors.hex2color(colors.cnames[name]) for name in cmap])
  return cmap[indices]


if __name__ == '__main__':
  from matplotlib import colors
  print(colors.hex2color(colors.cnames['orange']))
  cmap = np.array([[-2, -4], [2, 4.0]])
  indices = np.array([[0, 1], [1, 0]])
  im = cmap[indices]
  print(indices)
  print(im)

