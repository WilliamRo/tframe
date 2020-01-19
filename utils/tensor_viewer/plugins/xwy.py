from collections import OrderedDict

import numpy as np

from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView


"""
A perceptron operator (PO) can be described as 
                 y = \phi(x @ W + b).                                    (1)
Similarly, we use x and y to represent inputs and outputs to a PO..
We use w to represent values that relate to weights (can be weight matrix W, 
weight gradient or connection heat-map).

Say W in (1) is an M-by-N matrix, say 3x5, we have

                        w w w w w     
y y y y y  =  x x x  @  w w w w w  
                        w w w w w     

For display, x, w and y are organized as 

          y y y y y
        x w w w w w
        x w w w w w
        x w w w w w 
"""


def compose(x, w, y):
  # Sanity check
  assert isinstance(w, np.ndarray)
  assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
  x_dim, y_dim = x.size, y.size
  assert w.shape == (x_dim, y_dim)
  assert all([(t >= 0).all() for t in (x, w, y)])
  # Preprocess. Simply squash x, w, and y to be in [0, 1]
  x, w, y = [t / np.max(t) for t in (x, w, y)]

  # Initiate a blank matrix
  xwy = np.zeros(shape=[x_dim + 1, y_dim + 1], dtype=np.float)
  xwy[1:, 1:] = w
  xwy[1:, 0] = x.flatten()
  xwy[0, 1:] = y.flatten()

  # Return xwy
  return xwy

