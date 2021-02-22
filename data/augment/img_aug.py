from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tframe as tfr
from tframe.utils.arg_parser import Parser
from tframe.data.dataset import DataSet


def image_augmentation_processor(data_batch: DataSet, is_training: bool):
  # Get hub
  th = tfr.hub
  if not is_training or th.aug_config is None: return data_batch
  # Parse augmentation setting
  assert isinstance(th.aug_config, str)
  configs = [Parser.parse(s) for s in th.aug_config.split('|')]
  if len(configs) == 0: return data_batch

  # Apply each method according to configs
  for cfg in configs:
    # Find method
    if cfg.name == 'rotate': method = _rotate
    elif cfg.name == 'flip': method = _flip
    else: raise KeyError('!! Unknown augmentation option {}'.format(cfg.name))
    # Do augmentation
    data_batch.features = method(
      data_batch.features, *cfg.arg_list, **cfg.arg_dict)

  return data_batch


"""Currently this method works only for channel-last format.
   That is, the H and W dim of x correspond to x.shape[1] and x.shape[2].
   This applies to all the methods below.
"""

def _rotate(x: np.ndarray, bg=0):
  assert x.shape[1] == x.shape[2]
  return np.rot90(x, np.random.choice(4, 1)[0], axes=[1, 2])


def _flip(x: np.ndarray, bg=0):
  assert x.shape[1] == x.shape[2]
  for axis in (1, 2):
    if np.random.choice([True, False], 1)[0]: x = np.flip(x, axis=axis)
  return x
