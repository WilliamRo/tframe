import numpy as np


def bit_waves(num_bits, stack=False, stack_axis=-1):
  """This method generates n binary sequence of length 2^n, where n is the
     given `num_bits`. For example, if n = 3, 3 sequences will be generated:
     [[[1. 0. 1. 0. 1. 0. 1. 0.]],
      [[1. 1. 0. 0. 1. 1. 0. 0.]],
      [[1. 1. 1. 1. 0. 0. 0. 0.]]]
  """
  waves = []
  wave_let = np.array([[1.0, 0.0]], dtype=np.float32)
  for i in range(num_bits):
    waves.append(np.tile(wave_let, [1, 2 ** (num_bits - 1 - i)]))
    wave_let = np.transpose(wave_let)
    wave_let = np.reshape(np.concatenate([wave_let] * 2, axis=1), [1, -1])
  if stack: waves = np.stack(waves, axis=stack_axis)
  return waves


def expand_bit(array, axis):
  # Sanity check and get num_bits
  assert isinstance(array, np.ndarray)
  assert isinstance(axis, int) and axis < len(array.shape)
  n = array.shape[axis]

  # Expand array
  x = np.expand_dims(array, axis=axis)

  # Get waves, shape = (2^n, n)
  waves = bit_waves(n, stack=True, stack_axis=-1).squeeze()
  assert isinstance(waves, np.ndarray)
  new_shape = [1] * len(x.shape)
  new_shape[axis:axis+2] = waves.shape
  waves = waves.reshape(new_shape)

  # - Calculate bit_max
  y = waves * (1. - 2. * x) + x
  # now 1s in waves are replaced by x and 0s are replaced by 1 - x
  y = np.prod(y, axis=axis+1)
  return y


if __name__ == '__main__':
  axis = 2
  x = np.random.rand(2, 5, 2, 3)
  y = expand_bit(x, axis)
  print(y)
  print(np.sum(y, axis))
