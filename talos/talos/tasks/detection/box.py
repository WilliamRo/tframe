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
# ====-================================================================-========
"""This module provides some necessary APIs related to bounding boxes"""
import matplotlib.pyplot as plt

from talos import Nomear
from talos.tasks.detection.object2d import Object2D
from roma import check_type

import numpy as np



class Box(Object2D):

  def __init__(self, r_min, r_max, c_min, c_max, tag=None):
    """Given a 2-D image I, `self` represents a sub-region of
       I[r_min:r_max+1, c_min:c_max+1]"""
    assert r_min <= r_max and c_min <= c_max
    self.r_min = r_min
    self.r_max = r_max
    self.c_min = c_min
    self.c_max = c_max
    self.tag = check_type(tag, str, nullable=True)

  # region: Properties

  @property
  def height(self): return self.r_max - self.r_min + 1

  @property
  def width(self): return self.c_max - self.c_min + 1

  @property
  def yxhw(self): return self.r_min, self.c_min, self.height, self.width

  @property
  def xywh(self): return self.c_min, self.r_min, self.width, self.height

  @property
  def center(self):
    return (self.r_min + self.r_max + 1) / 2, (self.c_min + self.c_max + 1) / 2

  @property
  def area(self):
    return (self.r_max - self.r_min + 1) * (self.c_max - self.c_min + 1)

  def __str__(self):
    return f'[{self.r_min}:{self.r_max}, {self.c_min}:{self.c_max}]'

  # endregion: Properties

  # region: Public Methods

  def is_overlap_with(self, guest):
    assert isinstance(guest, Box)
    return (
        self.is_overlap_1D(self.r_min, self.r_max, guest.r_min, guest.r_max) and
        self.is_overlap_1D(self.c_min, self.c_max, guest.c_min, guest.c_max))

  def iou_to(self, guest):
    # Sanity check
    if isinstance(guest, list): guest = Box(*guest)
    assert isinstance(guest, Box)
    if not self.is_overlap_with(guest): return 0.0

    y1, y2 = self.intersect_1D(self.r_min, self.r_max, guest.r_min, guest.r_max)
    x1, x2 = self.intersect_1D(self.c_min, self.c_max, guest.c_min, guest.c_max)
    inter_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    return inter_area / (self.area + guest.area - inter_area)

  @staticmethod
  def iou(bound1, bound2):
    box1, box2 = Box(*bound1), Box(*bound2)
    return box1.iou_to(box2)

  @classmethod
  def get_grid_center_1D(cls, L: int, S: int, i: int):
    i_min, i_max = cls.get_cell_interval_1D(L, S, i)
    return i_min + (i_max - i_min + 1) / 2

  @classmethod
  def get_cell_interval_1D(cls, L, S, i, shift=0):
    """Return interval by pixel indices"""
    assert 0 <= i < S and -1 <= shift <= 1
    l = L // S  # min size
    i_min = i * l
    i_max = L - 1 if i == S - 1 else i_min + l - 1
    delta = shift * L
    return i_min + delta, i_max + delta

  # endregion: Public Methods

  # region: Plotter

  @staticmethod
  def show_rect_static(
      r_min, r_max, c_min, c_max, color='w', ax=None, margin=1):
    box = Box(r_min, r_max, c_min, c_max)
    box.show_rect(color=color, ax=ax, margin=margin)

  def show_rect(self, color='w', ax: plt.Axes = None, margin=1,
                show_tag=False, linestyle=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # If axes is not provided, get a default one
    if ax is None: ax = plt.gca()

    # Show rectangle
    x, y, w, h = self.xywh
    ax.add_patch(Rectangle((x - margin, y - margin), w + margin, h + margin,
                           edgecolor=color, facecolor='none', alpha=0.6,
                           linewidth=2, linestyle=linestyle))

    # Show tag if necessary
    if not show_tag: return
    ax.annotate(self.tag, (x, y), textcoords='offset points', xytext=(1, -9),
                ha='left', color='w', backgroundcolor='grey')

  # endregion: Plotter
