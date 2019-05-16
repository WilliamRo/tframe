from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

import tensorflow as tf

from tframe.data.base_classes import DataAgent


class TextDataAgent(DataAgent):


  @classmethod
  def check_level(cls, level):
    assert isinstance(level, str)
    level = level.lower()
    if level in ['w', 'word']: return 'word'
    elif level in ['c', 'char', 'character']: return 'char'
    else: raise ValueError('!! Can not resolve level `{}`'.format(level))


  @classmethod
  def read_txt(cls, file_path, split=False):
    py3 = sys.version_info[0] == 3
    with tf.gfile.GFile(file_path, "r") as f:
      if py3: text = f.read().replace("\n", "<eos>")
      else: text = f.read().decode("utf-8").replace("\n", "<eos>")
    if split: text = text.split()
    return text


  @classmethod
  def generate_mapping(cls, data):
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    tokens, _ = list(zip(*count_pairs))
    mapping = dict(zip(tokens, range(len(tokens))))
    return mapping


  @classmethod
  def generate_token_ids(cls, data, mapping):
    return [mapping[token] for token in data if token in mapping]

