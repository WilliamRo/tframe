from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def pre_emphasize(signal_, coefficient=0.97):
  assert isinstance(signal_, np.ndarray)
  return np.append(signal_[0], signal_[1:] - coefficient * signal_[:-1])
