from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def slerp(mu, low, high):
  """
  Spherical interpolation. Ref: Shoemake, 1985
  
  :param mu: scalar, \in [0, 1]
  :param low: vector with the same size of high
  :param high: vector
  :return: vector with the same size of low
  """
  # Check input
  if mu <= 0:
    return low
  elif mu >= 1:
    return high
  elif np.allclose(low, high):
    return low

  # Calculate theta
  q1 = low / np.linalg.norm(low)
  q2 = high / np.linalg.norm(high)
  theta = np.arccos(np.dot(q1, q2))

  sin = np.sin
  return sin((1 - mu)*theta)/sin(theta)*low + sin(mu*theta)/sin(theta)*high
