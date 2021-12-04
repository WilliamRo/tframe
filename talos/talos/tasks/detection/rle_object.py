# Copyright 2021 William Ro. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===-=======================================================================-==
"""This module provides a general class for run-length-encoded objects"""
from talos import Nomear
from talos.tasks.detection.object2d import Object2D
from roma import check_type

import numpy as np



class RLEObject(Object2D):

  def __init__(self, rle: list, width=None):
    # Sanity check
    check_type(rle, list, inner_type=int)
    assert len(rle) % 2 == 0
    self.rle = rle

    # Check and set width
    if width is not None: assert isinstance(width, int) and width > 1
    self.width = width

  # region: Properties

  @Nomear.property()
  def rle2(self):
    """For the i-th row, rle2[i] = [start_index, end_index]"""
    rle = np.reshape(self.rle, newshape=[-1, 2])
    rle[:, 1] = (rle[:, 0] + rle[:, 1] - 1)
    return rle

  @Nomear.property()
  def range1d(self): return min(self.rle2[:, 0]), max(self.rle2[:, 1])

  @Nomear.property()
  def rle3(self) -> np.ndarray:
    """For the i-th row, rle3[i] = [start_row, start_col, end_col]"""
    assert self.width is not None
    Rs = self.rle2 // self.width

    delta = Rs[:, 1] - Rs[:, 0]
    if sum(delta) != 0: raise ValueError('!! Length overflow detected')

    Cs = self.rle2 - Rs * self.width
    return  np.stack([Rs[:, 0], Cs[:, 0], Cs[:, 1]], axis=-1)

  @Nomear.property()
  def interval(self):
    return self.rle2[:, 0].min(), self.rle2[:, 1].max()

  @Nomear.property()
  def box(self):
    from talos.tasks.detection.box import Box
    r_min, r_max = self.rle3[:, 0].min(), self.rle3[:, 0].max()
    c_min, c_max = self.rle3[:, 1].min(), self.rle3[:, 2].max()
    assert r_max >= r_min and c_max >= c_min
    return Box(r_min, r_max, c_min, c_max)

  @Nomear.property()
  def mask2d(self):
    r1, r2, c1, c2 = (
      self.box.r_min, self.box.r_max, self.box.c_min, self.box.c_max)
    mask = np.zeros(shape=[r2-r1+1, c2-c1+1], dtype=bool)
    for r, c_1, c_2 in self.rle3: mask[r-r1, c_1-c1:c_2-c1+1] = 1
    return mask, (r1, c1)

  # endregion: Properties

  # region: Public Methods

  def is_overlap_with(self, guest):
    """Return whether the bounding box is overlapped with guest"""
    assert isinstance(guest, RLEObject)
    if self.width is None:
      min_1, max_1 = self.interval
      min_2, max_2 = guest.interval
      return self.is_overlap_1D(min_1, max_1, min_2, max_2)

    # For 2-D objects
    return self.box.is_overlap_with(guest.box)

  def iou_to(self, guest):
    # Sanity check
    if isinstance(guest, list): guest = RLEObject(guest)
    assert isinstance(guest, RLEObject)

    if not self.is_overlap_with(guest): return 0.0

    # Create minimum buffer
    (min1, max1), (min2, max2) = self.range1d, guest.range1d
    min_index = min(min1, min2)
    buf_len = max(max1, max2) - min_index + 1
    buffer = np.zeros(shape=[2, buf_len], dtype=bool)

    # Create masks
    for i, rle in enumerate((self.rle2, guest.rle2)):
      for a, b in rle: buffer[i][a-min_index:b-min_index+1] = True

    # Calculate score
    intersection = np.logical_and(buffer[0], buffer[1])
    union = np.logical_or(buffer[0], buffer[1])

    # Make sure the dominator is legal
    union_sum = np.sum(union)
    if union_sum == 0:
      raise AssertionError('!! Input RLEs for calculating IoU can not be empty')
    return np.sum(intersection) / union_sum

  @staticmethod
  def iou(rle_1: list, rle_2: list):
    rle_1, rle_2 = RLEObject(rle_1), RLEObject(rle_2)
    return rle_1.iou_to(rle_2)

  # endregion: Public Methods



if __name__ == '__main__':
  rle_1 = [3, 4, 10, 2]
  rle_2 = [2, 4, 11, 4]

  rle_1 = RLEObject(rle_1)

  print(rle_1.iou_to(rle_2))


