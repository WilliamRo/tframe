from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import tarfile
import collections

import numpy as np
import tensorflow as tf
from tframe import console

# from tframe.data.dataset import DataSet
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.base_classes import DataAgent
# from tframe.data.sequences.nlp.text_data_agent import TextDataAgent


class IMDB(DataAgent):
  """The IMDB movie review data set.

     References
     [1] Maas, Andrew L. etc. Learning Word Vectors for Sentiment Analysis. 2011
     [2] MGU16
  """

  DATA_NAME = 'IMDB'
  DATA_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

  @classmethod
  def load(cls, data_dir, **kwargs):
    data_set = cls.load_as_tframe_data(data_dir)
    return data_set

  @classmethod
  def load_as_tframe_data(cls, data_dir, **kwargs):
    # Load directly if data set exists
    data_path = cls._get_data_path(data_dir)
    if os.path.exists(data_path): seq_set = SequenceSet.load(data_path)
    else:
      # If data does not exist, create from raw data
      console.show_status('Creating data sets ...')
      raw_data = cls._load_raw_data(data_dir)

      seq_set = None

    return seq_set

  # region : Private Methods

  @classmethod
  def _load_raw_data(cls, data_dir):
    # Check .feat files
    feat_paths = [os.path.join(data_dir, 'aclImdb', dir_name, 'labeledBow.feat')
                  for dir_name in ('train', 'test')]
    # Download and unzip if necessary
    if not all([os.path.exists(p) for p in feat_paths]):
      # Check gz file, download if necessary
      gz_file_path = cls._check_raw_data(data_dir)
      # Extract file to data_dir
      gz_file_name = os.path.basename(gz_file_path)
      console.show_status('Extracting {} ...'.format(gz_file_name))
      tarfile.open(gz_file_path).extractall(data_dir)

    # Read .feat files
    # TODO

    return None

  @classmethod
  def _get_data_path(cls, data_dir):
    return os.path.join(data_dir, 'IMDB.tfds')

  # endregion : Private Methods
