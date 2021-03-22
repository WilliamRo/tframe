from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import tframe as tfr


"""Stark knows everything about tframe logic and helps to mitigate the chaos 
   in this framework.
"""
# region: Pruner-related

def get_params_num(variables, consider_prune=False):
  """Used in Net.structure_detail"""
  if len(variables) == 0:
    if consider_prune: return 0, 0
    else: return 0
  tfr.checker.check_type(variables, tf.Variable)
  if not isinstance(variables, (list, tuple)): variables = [variables]
  num, dense_num = 0, 0
  for v in variables:
    n, dn = _get_params_num_single(v, True)
    num, dense_num = num + n, dense_num + dn
  # Return according to consider_prune
  if consider_prune: return num, dense_num
  else: return num

def _get_params_num_single(variable, consider_prune=False):
  assert isinstance(variable, tf.Variable)

  pruner = tfr.context.pruner
  if all([consider_prune, pruner is not None]):
    if variable in pruner.variable_dict:
      return pruner.get_variable_sizes(variable)

  # Compute variable size by default method
  size = int(np.prod(variable.shape))
  # Return according to consider_prune
  if consider_prune: return size, size
  else: return size

def get_num_string(num, dense_num):
  if num == 0: num_str = ''
  elif tfr.hub.prune_on or tfr.hub.etch_on:
    num_str = '{} ({:.1f}%)'.format(num, 100.0 * num / dense_num)
  else: num_str = str(num)
  return num_str

# endregion: Pruner-related

# region: MISC

def decayable(v):
  assert isinstance(v, tf.Variable)
  return all([v.trainable, 'batchnorm' not in v.name, 'bias' not in v.name])

# endregion: MISC

