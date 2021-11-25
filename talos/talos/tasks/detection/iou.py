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
# ==-==========================================================================-
"""Similar functions can be found in sklearn.metrics.jaccard_score"""
import numpy as np
from typing import Union, Optional



def iou(rle_1: Union[list, np.ndarray], rle_2: Union[list, np.ndarray]):
  """Calculate intersection over union (IoU) score.

  Both inputs should be of type
  (1) list/np.ndarray of run length encoding (RLE) format, or
  (2) np.ndarray of RLE with shape (?, 2)

  :return: IoU score
  """
  # Sanity check
  assert type(rle_1) is type(rle_2)

  # Convert to (?, 2)-shape array if necessary
  if isinstance(rle_1, list) or len(rle_1.shape) == 1:
    assert len(rle_1) % 2 == 0 and len(rle_2) % 2 == 0
    rle_1, rle_2 = [np.reshape(rle, (-1, 2)) for rle in (rle_1, rle_2)]

  rles = [np.stack([rle[:, 0], rle[:, 0] + rle[:, 1]], axis=-1)
          for rle in (rle_1, rle_2)]

  # Calculate left bound and shift each rle
  min_index = min([min(rle[:, 0]) for rle in rles])
  rles = [rle - min_index for rle in rles]
  max_index = max([max(rle[:, 1]) for rle in rles])

  # Create an empty buffer
  buffer = np.zeros(shape=[2, max_index], dtype=bool)

  # Create masks
  for i, rle in enumerate(rles):
    for a, b in rle: buffer[i][a:b] = True

  # Calculate score
  intersection = np.logical_and(buffer[0], buffer[1])
  union = np.logical_or(buffer[0], buffer[1])
  return np.sum(intersection) / np.sum(union)



if __name__ == '__main__':
  rle_1 = [3, 4, 10, 2]
  rle_2 = [2, 4, 11, 4]
  print(iou(rle_1, rle_2))
