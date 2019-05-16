from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile

import numpy as np
from tframe import console

from tframe.data.dataset import DataSet
from tframe.data.sequences.nlp.text_data_agent import TextDataAgent


class Text8(TextDataAgent):
  """Text8 data set
     Reference: http://mattmahoney.net/dc/textdata.html
  """

  DATA_NAME = 'Text8'
  DATA_URL = 'http://mattmahoney.net/dc/text8.zip'

  @classmethod
  def load(cls, data_dir, train_size=-1, val_size=5000000, test_size=4999999,
           **kwargs):
    data_set = cls.load_as_tframe_data(data_dir)
    console.show_status('Splitting ...')
    return cls._split_and_return(data_set, train_size, val_size, test_size)

  @classmethod
  def load_as_tframe_data(cls, data_dir, **kwargs):

    # Load directly if all files exists
    data_path = cls._get_data_paths(data_dir)
    if os.path.exists(data_path):
      data_set = DataSet.load(data_path)
    else:
      # If data does not exist, create from raw data
      console.show_status('Creating data sets ...')
      data, mapping = cls._load_raw_data(data_dir)
      x = np.array(data[:-1]).reshape(-1, 1)
      y = np.array(data[1:]).reshape(-1, 1)
      data_set = DataSet(x, y, name='Text8.char', mapping=mapping)
      # Save data set and show info
      data_set.save(data_path)
      console.show_status('{} saved to `{}`'.format(data_set.name, data_path))

    # Show mapping size
    console.show_status(
      'Data sets (containing {} different characters) loaded:'.format(
      len(data_set['mapping'])))

    return data_set

  # region : Private Methods

  @classmethod
  def _get_data_paths(cls, data_dir):
    return os.path.join(data_dir, 'TEXT8.char.tfd')

  @classmethod
  def _load_raw_data(cls, data_dir):
    # Check raw file path
    file_path = os.path.join(data_dir, 'text8')
    # Download and unzip if necessary
    if not os.path.exists(file_path):
      # Check zip file, download if necessary
      zip_file_path = cls._check_raw_data(data_dir)
      # Extract file to data_dir
      zip_file_name = os.path.basename(zip_file_path)
      console.show_status('Extracting {} ...'.format(zip_file_name))
      zipfile.ZipFile(zip_file_path, 'r').extractall(data_dir)

    # - Generate data from .txt files
    text = cls.read_txt(file_path, split=False)
    mapping = cls.generate_mapping(text)
    data = cls.generate_token_ids(text, mapping)

    return data, mapping

  # endregion : Private Methods



