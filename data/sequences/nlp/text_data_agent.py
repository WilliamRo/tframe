from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys, os

from tframe import tf

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


  @classmethod
  def read_signal_as_text(cls, file_path, vocab=None) -> (list, dict):
    if not os.path.exists(file_path):
      raise FileNotFoundError(f'!! File `{file_path}` not found.')

    # Read text
    with open(file_path, 'rb') as f: text = f.read()

    # Generate vocabulary set if not provided
    if vocab is None: vocab = sorted(set(text))

    # Create mapping function
    mapping = {c:i for i, c in enumerate(vocab)}

    # Return list of indices and vocab
    return [mapping[c] for c in text], mapping


