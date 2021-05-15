from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import colors


def tile_kernels(kernel: np.ndarray, axis=-1):
  # input shape of x should be [K, K, I, O]
  assert len(kernel.shape) == 4 and kernel.shape[0] == kernel.shape[1]
  assert axis in (-1, -2, 2, 3)
  axis_kept = 2 if axis in (-1, 3) else 3
  K = kernel.shape[0]
  N = int(np.ceil(np.sqrt(kernel.shape[axis])))
  # Calculate shape
  L = N * (K + 1) - 1
  F = np.zeros(shape=[L, L, kernel.shape[axis_kept]], dtype=np.float)

  for k in range(kernel.shape[axis_kept]):
    for i in range(N):
      for j in range(N):
        index = i * N + j
        if index >= kernel.shape[axis]: break
        x, y = i * (K + 1), j * (K + 1)
        if axis_kept == 2: F[x:x+K, y:y+K, k] = kernel[:, :, k, index]
        else: F[x:x+K, y:y+K, k] = kernel[:, :, index, k]
  return F


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

