from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import hub as th
from tframe import monitor


from .etch_kernel import EtchKernel


class Eraser(EtchKernel):
  """The second official etch  kernel.
  """

  def __init__(self, weights):
    # Call parent's constructor
    super().__init__(weights)


  def _get_new_mask(self):
    pre_mask = self.mask_buffer

    monitor

    return pre_mask



