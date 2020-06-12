import numpy as np


def get_ravel_indices(a):
  assert isinstance(a, np.ndarray)
  indices = np.meshgrid(*[range(s) for s in a.shape], indexing='ij')
  return tuple([v.ravel() for v in indices])


if __name__ == '__main__':
  a = np.arange(24).reshape(2, 3, 4)
  assert isinstance(a, np.ndarray)
  indices = get_ravel_indices(a)
  delta = np.linalg.norm(a[indices] - a.ravel())
  print('Delta = {}'.format(delta))

