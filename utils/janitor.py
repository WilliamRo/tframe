from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def wrap(obj, obj_type=None, wrap_as=list):
  """Wrap obj into list."""
  assert wrap_as in (list, tuple)
  if not isinstance(obj, wrap_as): obj = wrap_as([obj])
  if obj_type is not None:
    from tframe import checker
    obj = checker.check_type_v2(obj, obj_type)
  return obj


def recover_seq_set_outputs(outputs, seq_set):
  """Outputs of tframe batch evaluation are messed up.
     This method will help.
     Currently only full size evaluation is supported
  """
  from tframe.data.sequences.seq_set import SequenceSet

  assert isinstance(outputs, list) and isinstance(seq_set, SequenceSet)

  # Recover
  results = [[] for _ in range(seq_set.size)]
  remaining = [l for l in seq_set.structure]
  indices = list(range(seq_set.size))
  while indices:
    retire_list = []
    for i in indices:
      output = outputs.pop(0)
      results[i].append(output)
      remaining[i] -= len(output)
      if remaining[i] == 0: retire_list.append(i)
    for i in retire_list: indices.remove(i)
  # Concatenate results
  results = [np.concatenate(array_list) for array_list in results]
  assert [len(a) for a in results] == seq_set.structure

  return results


