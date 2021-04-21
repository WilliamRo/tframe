from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tframe as tfr
from tframe.utils.arg_parser import Parser
from tframe.data.dataset import DataSet

from typing import Optional


def image_augmentation_processor(
    data_batch: DataSet, is_training: bool, proceed_target: bool = False):
  # Get hub
  th = tfr.hub
  if not is_training or th.aug_config is None: return data_batch
  # Parse augmentation setting
  assert isinstance(th.aug_config, str)
  if th.aug_config in ('-', 'x'): return data_batch
  configs = [Parser.parse(s) for s in th.aug_config.split('|')]
  if len(configs) == 0: return data_batch

  # Apply each method according to configs
  for cfg in configs:
    # Find method
    if cfg.name == 'rotate': method = _rotate
    elif cfg.name == 'flip': method = _flip
    else: raise KeyError('!! Unknown augmentation option {}'.format(cfg.name))
    # Do augmentation
    if proceed_target:
      data_batch.features, data_batch.targets = method(
        data_batch.features, data_batch.targets, *cfg.arg_list, **cfg.arg_dict)
    else: data_batch.features = method(
      data_batch.features, *cfg.arg_list, **cfg.arg_dict)

  return data_batch


"""Currently this method works only for channel-last format.
   That is, the H and W dim of x correspond to x.shape[1] and x.shape[2].
   This applies to all the methods below.
"""

def _rotate(x: np.ndarray, y: Optional[np.ndarray] = None, bg=0):
  # Check x shape
  assert x.shape[1] == x.shape[2]
  # Decide k
  k = np.random.choice(4, 1)[0]
  # Rotate x
  x = np.rot90(x, k, axes=[1, 2])

  if y is not None:
    assert y.shape[1] == y.shape[2]
    y = np.rot90(y, k, axes=[1, 2])
    return x, y

  return x


def _flip(x: np.ndarray, horizontal=True, vertical=True, p=0.5,
          y: Optional[np.ndarray] = None):
  """Randomly flip image batch.

  :param x: images with shape (batch_size, H, W[, C])
  :param horizontal: whether to do flip horizontally
  :param vertical: whether to do flip vertically
  :param p: probability to do flip
  :return: flipped image batch
  """
  assert 0 < p < 1 and any([horizontal, vertical])

  def _rand_flip(axis):
    mask = np.random.choice([True, False], size=x.shape[0], p=[p, 1 - p])
    x[mask] = np.flip(x[mask], axis=axis)
    if y is not None: y[mask] = np.flip(y[mask], axis=axis)

  if horizontal: _rand_flip(2)
  if vertical: _rand_flip(1)

  if y is not None: return x, y
  return x
