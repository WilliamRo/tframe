from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

import numpy as np
from tframe import console

from tframe.data.dataset import DataSet
from tframe.data.sequences.nlp.text_data_agent import TextDataAgent


class PTB(TextDataAgent):
  """Penn Tree Bank dataset"""

  DATA_NAME = 'PTB'
  DATA_URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'

  @classmethod
  def load(cls, data_dir, level, **kwargs):
    return cls.load_as_tframe_data(data_dir, level)

  @classmethod
  def load_as_tframe_data(cls, data_dir, level, **kwargs):
    # Check level
    level = cls.check_level(level)
    # Load directly if all files exists
    data_paths = cls._get_data_paths(data_dir, level)
    if all([os.path.exists(path) for path in data_paths]):
      data_sets = [DataSet.load(path) for path in data_paths]
    else:
      # If data does not exist, create from raw data
      console.show_status('Creating data sets ...')
      raw_data = cls._load_raw_data(data_dir, level)

      train_data, valid_data, test_data, mapping = raw_data
      data_sets = []
      for data, name, path in zip(
          raw_data[:3], ('Train Set', 'Valid Set', 'Test Set'), data_paths):
        x = np.array(data[:-1]).reshape(-1, 1)
        y = np.array(data[1:]).reshape(-1, 1)
        data_set = DataSet(x, y, name=name, mapping=mapping)
        data_set.save(path)
        console.show_status('{} saved to `{}`'.format(name, path))
        data_sets.append(data_set)

    # Show mapping size
    console.show_status('Data sets (containing {} {}s) loaded:'.format(
      len(data_sets[0]['mapping']), level))
    for data_set in data_sets:
      assert isinstance(data_set, DataSet)
      console.supplement('Length of {} is {}'.format(
        data_set.name, data_set.size))
    return data_sets[0], data_sets[1], data_sets[2]

  # region : Private Methods

  @classmethod
  def _get_data_paths(cls, data_dir, level):
    level = cls.check_level(level)
    data_path_format = os.path.join(data_dir, 'PTB.' + level + '.{}.tfd')
    return [data_path_format.format(s) for s in ('train', 'valid', 'test')]

  @classmethod
  def _load_raw_data(cls, data_dir, level):
    # Check tgz file, download if necessary
    tgz_file_path = cls._check_raw_data(data_dir)
    # Extract file to data_dir
    tgz_file_name = os.path.basename(tgz_file_path)
    console.show_status('Extracting {} ...'.format(tgz_file_name))
    tarfile.open(tgz_file_path).extractall(data_dir)

    # Find train, valid and test file paths
    folder_name = tgz_file_name.split('.')[0]
    data_dir = os.path.join(data_dir, folder_name, 'data')
    level = cls.check_level(level)
    level_str = 'char.' if level == 'char' else ''
    file_format = 'ptb.' + level_str + '{}.txt'
    file_paths = []
    for s in ['train', 'valid', 'test']:
      file_paths.append(os.path.join(data_dir, file_format.format(s)))

    # Read tokens
    train_tokens, valid_tokens, test_tokens = [
      cls.read_txt(path, split=True) for path in file_paths]
    # Build vocabulary using train tokens
    mapping = cls.generate_mapping(train_tokens)
    # Get ids
    train_data = cls.generate_token_ids(train_tokens, mapping)
    valid_data = cls.generate_token_ids(valid_tokens, mapping)
    test_data = cls.generate_token_ids(test_tokens, mapping)

    return train_data, valid_data, test_data, mapping

  # endregion : Private Methods



