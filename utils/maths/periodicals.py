import numpy as np


def bit_waves(num_bits, stack=False, axis=-1):
  waves = []
  wave_let = np.array([[1.0, 0.0]], dtype=np.float32)
  for i in range(num_bits):
    waves.append(np.tile(wave_let, [1, 2 ** (num_bits - 1 - i)]))
    wave_let = np.transpose(wave_let)
    wave_let = np.reshape(np.concatenate([wave_let] * 2, axis=1), [1, -1])
  if stack: waves = np.stack(waves, axis=axis)
  return waves


if __name__ == '__main__':
  waves = bit_waves(3)
  for w in waves: print(w)
