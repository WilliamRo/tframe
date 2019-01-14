from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import checker


def pre_emphasize(signal_, coefficient=0.97):
  assert isinstance(signal_, np.ndarray)
  return np.append(signal_[0], signal_[1:] - coefficient * signal_[:-1])


def short_time_energy(signal_, window_size, stride, center=True):
  """Calculate short time energy
  :param center: If `True`, the signal `y` is padded so that frame
                 `D[:, t]` is centered at `y[t * hop_length]`.
                 If `False`, then `D[:, t]` begins at `y[t * hop_length]`
  """

  # Sanity check
  assert isinstance(signal_, np.ndarray)
  checker.check_positive_integer(window_size)
  checker.check_positive_integer(stride)
  if center:
    signal_ = np.pad(signal_, window_size // 2, mode='reflect')
  # Reshape signal
  signal_ = np.reshape(signal_, newshape=(-1,))
  # Form causal matrix
  frames = []
  cursor = 0
  while True:
    frames.append(signal_[cursor : cursor + window_size])
    # Move cursor forward
    cursor += stride
    if cursor + window_size > signal_.size: break
  # Calculate energy
  stack = np.stack(frames, axis=0)
  energy = np.multiply(stack, stack)
  energy = np.sum(energy, axis=1)
  return energy
