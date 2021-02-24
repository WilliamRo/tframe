import numpy as np


def get_ravel_indices(a):
  assert isinstance(a, np.ndarray)
  indices = np.meshgrid(*[range(s) for s in a.shape], indexing='ij')
  return tuple([v.ravel() for v in indices])


def pad_or_crop(a, axis, size, crop_mode='center',
                pad_mode='constant', constant_val=0):
  assert isinstance(a, np.ndarray)
  assert isinstance(axis, int) and 0 <= axis
  assert isinstance(size, int) and size > 0
  # Currently for crop, only 'center' mode is supported
  assert crop_mode == 'center'

  # Get array shape
  shape = a.shape
  assert len(shape) > axis
  if size == shape[axis]: return a
  elif size < shape[axis]:
    # Crop
    start_i = (shape[axis] - size) // 2
    end_i = start_i + size
    if axis == 0: return a[start_i:end_i]
    elif axis == 1: return a[:, start_i:end_i]
    elif axis == 2: return a[:, :, start_i:end_i]
    else: raise NotImplemented('!! axis > 2 is not supported for crop')
  else:
    # Pad
    pw1 = (size - shape[axis]) // 2
    pw2 = size - shape[axis] - pw1
    pad_width = [[0, 0] for _ in shape]
    pad_width[axis] = [pw1, pw2]
    configs = {
      'constant_values': constant_val} if pad_mode == 'constant' else {}
    return np.pad(a, pad_width, mode=pad_mode, **configs)


if __name__ == '__main__':
  a = np.arange(25).reshape(5, 5)
  print(a)
  a = pad_or_crop(a, 0, 7)
  a = pad_or_crop(a, 1, 3)
  print(a)

