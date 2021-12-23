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
# ===-==================================================================-=======
"""This module provides a base class for 2D objects"""
from talos import Nomear
from roma import check_type

import numpy as np



class Object2D(Nomear):

  @staticmethod
  def is_overlap_1D(min_1, max_1, min_2, max_2):
    return max_2 >= min_1 and max_1 >= min_2

  @staticmethod
  def intersect_1D(min_1, max_1, min_2, max_2):
    assert Object2D.is_overlap_1D(min_1, max_1, min_2, max_2)
    return max(min_1, min_2), min(max_1, max_2)

  # region: APIs

  def is_overlap_with(self, guest): raise NotImplementedError

  def iou_to(self, guest): raise NotImplementedError

  @staticmethod
  def iou(obj1, obj2): raise NotImplementedError

  # endregion: APIs

  # region: Metrics

  @classmethod
  def calc_match_matrix(cls, true_list, pred_list):
    """Calculate the match matrix for one image m of shape [M, N], where
       M = len(true_list), N = len(pred_list), and
       m[i, j] = iou(true_list[i], pred_list[j])
    """
    # Sanity check
    check_type(true_list, list, inner_type=Object2D)
    check_type(pred_list, list, inner_type=Object2D)

    # Create the matrix
    M, N = len(true_list), len(pred_list)
    m = np.zeros(shape=[M, N], dtype=float)
    for i, true_obj in enumerate(true_list):
      for j, pred_obj in enumerate(pred_list):
        m[i, j] = true_obj.iou_to(pred_obj)

    return m

  @classmethod
  def calc_avg_precision(cls, true_list, pred_list, thresholds,
                         multiply_confidence=False):
    """Calculate the average precision (AP) metric at a given threshold
       or a set of thresholds"""
    # Sanity check
    check_type(true_list, list, inner_type=Object2D)
    check_type(pred_list, list, inner_type=Object2D)
    if not isinstance(thresholds, (list, tuple, set, np.ndarray)):
      thresholds = [thresholds]

    # Sweep over all thresholds
    m = cls.calc_match_matrix(true_list, pred_list)
    precisions = []
    for alpha in thresholds:
      acc_pred = np.minimum(np.sum(m >= alpha, axis=0), 1.0)

      if multiply_confidence:
        assert hasattr(pred_list[0], 'confidence')
        c = np.array([p.confidence for p in pred_list])
        TP = sum(acc_pred * c)
      else: TP = sum(acc_pred)

      precisions.append(TP / len(pred_list))

    # Return AP
    if len(precisions) == 1: return precisions[0]
    return np.average(precisions)

  # endregion: Metrics
