from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import numpy as np
from tframe import console

from tframe.data.dataset import DataSet
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.base_classes import DataAgent

from tframe.utils.display.progress_bar import ProgressBar
from tframe.utils.display.table import Table

from tframe import checker
from tframe import Classifier
from tframe.layers.layer import Layer, single_input
from tframe.trainers.trainer import Trainer
from tframe import hub as th


class FI2010(DataAgent):
  """
  A LOB-ITCH dataset extracted from five stocks traded on the NASDAQ OMX Nordic
  at the Helsinki exchange from 1 June 2010 to 14 June 2010.
  -----------------------------------------------------------------------------
  Id     ISIN Code     Company(Oyj)  Sector              Industry
  -----------------------------------------------------------------------------
  KESBV  FI0009000202  Kesko         Consumer Defensive  Grocery Stores
  OUT1V  FI0009002422  Outokumpu     Basic Materials     Steel
  SAMPO  FI0009003305  Sampo         Financial Services  Insurance
  RTRKS  FI0009003552  Rautaruukki   Basic Materials     Steel
  WRT1V  FI0009000727  Wartsila      Industrials         Diversiﬁed Industrials
  -----------------------------------------------------------------------------
  Day       1      2      3      4      5      6      7      8      9     10
  Blocks  39512  38397  28535  37023  34785  39152  37346  55478  52172  31937

  Reference:
  [1] Adamantios Ntakaris, Martin Magris, Juho Kanniainen, Moncef Gabbouj
      and Alexandros Iosifidis. Benchmark Dataset for Mid-Price Forecasting of
      Limit Order Book Data with Machine Learning Methods. 2018.
  """

  DATA_NAME = 'FI-2010'
  DATA_URL = 'https://etsin.fairdata.fi/api/dl?cr_id=73eb48d7-4dbc-4a10-a52a-da745b47a649&file_id=5b32ac028ab4d130110888f19872320'

  @classmethod
  def load(cls, data_dir, auction=False, norm_type='zscore', setup=2,
           val_size=None, horizon=100, **kwargs):
    # Sanity check
    assert setup in [1, 2]
    # Load tframe data
    seq_set = cls.load_as_tframe_data(
      data_dir, auction=auction, norm_type=norm_type, setup=setup)
    seq_set = cls.extract_seq_set(seq_set, horizon)
    # Separate dataset according to setup
    if setup == 2:
      train_set, test_set = seq_set.split(1, 3, names=('Train Set', 'Test Set'))
      assert isinstance(train_set, SequenceSet)
      train_set = train_set.stack
      if val_size == 0: return train_set, test_set
      if val_size is None: val_size = 54750
      assert isinstance(val_size, int) and val_size > 0
      train_set, val_set = train_set.split(
        -1, val_size, names=('Train Set', 'Val Set'))
      return train_set, val_set, test_set
    else: raise NotImplementedError

  @classmethod
  def extract_seq_set(cls, raw_set, horizon):
    assert isinstance(raw_set, SequenceSet) and horizon in [10, 20, 30, 50, 100]
    seq_set = SequenceSet(
      features=[array[:, :40] for array in raw_set.data_dict['raw_data']],
      targets=raw_set.data_dict[horizon],
      name=raw_set.name,
    )
    return seq_set

  @classmethod
  def load_as_tframe_data(
      cls, data_dir, auction=False, norm_type='zscore', setup=2, **kwargs):
    # Confirm type of normalization
    nt_lower = norm_type.lower()
    # 'Zscore' for directory names and 'ZScore' for file names
    if nt_lower in ['1', 'zscore']: type_id, norm_type = 1, 'Zscore'
    elif nt_lower in ['2', 'minmax']: type_id, norm_type = 2, 'MinMax'
    elif nt_lower in ['3', 'decpre']: type_id, norm_type = 3, 'DecPre'
    else: raise KeyError('Unknown type of normalization `{}`'.format(norm_type))
    # Load directly if dataset exists
    data_path = cls._get_data_path(data_dir, auction, norm_type, setup)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If dataset does not exist, create from raw data
    console.show_status('Creating `{}` from raw data ...'.format(
      os.path.basename(data_path)))
    # Load raw data
    features, targets = cls._load_raw_data(
      data_dir, auction, norm_type, type_id, setup)

    # Wrap raw data into tframe Sequence set
    data_dict = {'raw_data': features}
    data_dict.update(targets)
    seq_set = SequenceSet(data_dict=data_dict, name=cls.DATA_NAME)
    # Save Sequence set
    seq_set.save(data_path)
    console.show_status('Sequence set saved to `{}`'.format(data_path))
    # Return
    return seq_set

  # region : Private Methods

  @classmethod
  def _load_raw_data(cls, data_dir, auction, norm_type, type_id, setup):
    assert isinstance(auction, bool)
    if not isinstance(norm_type, str): norm_type = str(norm_type)
    # Confirm sub-directory name
    auction_dir_name = 'Auction'
    if not auction: auction_dir_name = 'No' + auction_dir_name
    # Get directory name for training and test set
    norm_dir_name = '{}.{}_{}'.format(type_id, auction_dir_name, norm_type)
    path = os.path.join(
      data_dir, 'BenchmarkDatasets', auction_dir_name, norm_dir_name)
    training_set_path = os.path.join(
      path, '{}_{}_Training'.format(auction_dir_name, norm_type))
    test_set_path = os.path.join(
      path, '{}_{}_Testing'.format(auction_dir_name, norm_type))

    # Check training and test path
    if any([not os.path.exists(training_set_path),
            not os.path.exists(test_set_path)]):
      import zipfile
      zip_file_name = 'BenchmarkDatasets.zip'
      zip_file_path = cls._check_raw_data(data_dir, zip_file_name)
      console.show_status(
        'Extracting {} (this may need several minutes) ...'.format(zip_file_name))
      zipfile.ZipFile(zip_file_path, 'r').extractall(data_dir)
      console.show_status('{} extracted successfully.'.format(data_dir))
    assert all(
      [os.path.exists(training_set_path), os.path.exists(test_set_path)])

    # Read data and return
    return cls._read_10_days(
      training_set_path, test_set_path, auction, norm_type, setup)

  @classmethod
  def _get_data_file_path_list(cls, dir_name, training, auction, norm_type):
    if norm_type == 'Zscore': norm_type = 'ZScore'
    auction = 'Auction' if auction else 'NoAuction'
    prefix = 'Train' if training else 'Test'
    file_path_list = [os.path.join(dir_name, '{}_Dst_{}_{}_CF_{}.txt'.format(
      prefix, auction, norm_type, i)) for i in range(1, 10)]
    # Make sure each file exists
    for file_path in file_path_list:
      if not os.path.exists(file_path):
        raise FileExistsError('File `{}` not found.'.format(file_path))
    return file_path_list

  @classmethod
  def _read_10_days(cls, train_dir, test_dir, auction, norm_type, setup):
    """Read train_1, test_1, ... test_9 in order."""
    assert setup == 2
    train_paths = cls._get_data_file_path_list(
      train_dir, True, auction, norm_type)
    test_paths = cls._get_data_file_path_list(
      test_dir, False, auction, norm_type)
    # Read data from .txt files
    features, targets = [], {}
    horizons = [10, 20, 30, 50, 100]
    for h in horizons: targets[h] = []
    dim = 144
    # For Setup 2, train_set = train_7, test_set = test_[789]
    data_paths = [train_paths[6]] + test_paths[-3:]
    for i, path in enumerate(data_paths):
      # Create day slot for features and targets
      features.append([])
      for h in horizons: targets[h].append([])
      console.show_status('Reading data from `{}` ...'.format(
        os.path.basename(path)))
      with open(path, 'r') as f: lines = f.readlines()
      # Sanity check
      assert len(lines) == dim + len(horizons)
      # Parse data
      data = [[s for s in line.split(' ') if s] for line in lines]
      total = len(data[0])
      assert [len(d) == total for d in data]
      bar = ProgressBar(total)
      # Put data appropriately
      for j, str_list in enumerate(zip(*data)):
        col = np.array(str_list, dtype=np.float)
        features[-1].append(col[:dim])
        for k, h in enumerate(horizons): targets[h][-1].append(col[dim + k])
        # Refresh progress bar
        bar.show(j + 1)
      # Stack list
      features[-1] = np.stack(features[-1], axis=0)
      for k, h in enumerate(horizons):
        targets[h][-1] = np.array(
          np.stack(targets[h][-1], axis=0), dtype=np.int64) - 1
      console.show_status('Successfully read {} event blocks'.format(total))
    # Sanity check and return
    total = sum([len(x) for x in features])
    console.show_status('Totally {} event blocks read.'.format(total))
    if not auction: assert total == 394337
    else: assert total == 458125
    return features, targets

  @classmethod
  def _get_data_path(cls, data_dir, auction, norm_type, setup):
    assert isinstance(auction, bool)
    file_name = 'FI-2010-{}Auction-{}-Setup{}.tfds'.format(
     '' if auction else 'No', norm_type, setup)
    return os.path.join(data_dir, file_name)

  # endregion : Private Methods

  # region : Probe and evaluate

  @staticmethod
  def probe(dataset, trainer):
    from tframe.trainers.trainer import Trainer
    # Sanity check
    assert isinstance(trainer, Trainer) and isinstance(dataset, DataSet)
    model = trainer.model
    assert isinstance(model, Classifier)
    # Make prediction
    if 'order002' in th.developer_code:
      table, _ = FI2010._get_stats(model, dataset, batch_size=3)
    else: table, _ = FI2010._get_stats(model, dataset[:4000])
    return table.content

  @staticmethod
  def evaluate(entity, seq_set):
    is_training = isinstance(entity, Trainer)
    if is_training: model = entity.model
    else: model = entity
    # Sanity check
    assert isinstance(model, Classifier) and isinstance(seq_set, SequenceSet)
    # Get table and F1 score
    table, F1 = FI2010._get_stats(model, seq_set, batch_size=seq_set.size)
    table.print_buffer()
    # Take note if is training
    if is_training:
      model.agent.take_notes(table.content)
      model.agent.put_down_criterion('F1 Score', F1)

  @staticmethod
  def _get_stats(model, dataset, batch_size=1):
    # Sanity check
    assert isinstance(model, Classifier)
    # Get predictions and labels
    label_pred_tensor = model.key_metric.quantity_definition.quantities
    label_pred = model.evaluate(
      label_pred_tensor, dataset, batch_size=batch_size, verbose=True)
    console.show_status('Evaluation completed')
    label_pred = np.concatenate(label_pred, axis=0)
    # TODO: here results for each day can be divided wisely
    # Initialize table
    movements = ['Upward', 'Stationary', 'Downward']
    header = ['Movement  ', 'Accuracy %', 'Precision %', 'Recall %',
              '   F1 %']
    widths = [len(h) for h in header]
    table = Table(*widths, tab=3, margin=1, buffered=True)
    table.specify_format(*['{:.2f}' for _ in header], align='lrrrr')
    table.hdash()
    table.print('Prediction Horizon k = {}'.format(th.horizon))
    table.print_header(*header)
    # Get statistics
    precisions, recalls, F1s = [], [], []
    x = label_pred
    for c, move in enumerate(movements):
      col, row = x[x[:, 0] == c][:, 1], x[x[:, 1] == c][:, 0]
      TP = len(col[col == c])
      FP, FN = len(row) - TP, len(col) - TP
      precision = TP / (TP + FP) * 100 if TP + FP > 0 else 0
      recall = TP / (TP + FN) * 100 if TP+ FN > 0 else 0
      if precision + recall == 0: F1 = 0
      else: F1 = 2 * precision * recall / (precision + recall)
      precisions.append(precision)
      recalls.append(recall)
      F1s.append(F1)
      # Show in table
      table.print_row(move, '-', precision, recall, F1)
    precision, recall = np.mean(precisions), np.mean(recalls)
    F1, accuracy = np.mean(F1s), 100 * np.sum(x[:, 0] == x[:, 1]) / len(x)
    # Print and save
    table.hline()
    table.print_row('Overall', accuracy, precision, recall, F1)
    table.hline()

    return table, F1

  # endregion : Probe and evaluate


class Extract(Layer):

  full_name = 'extract'
  abbreviation = 'extract'

  def __init__(self):
    self._max_level = checker.check_positive_integer(th.max_level)
    assert self._max_level <= 10
    self._volume_only = checker.check_type(th.volume_only, bool)
    self.output_scale = self._max_level * (2 if self._volume_only else 4)

  @single_input
  def _link(self, x, **kwargs):
    """x = {P_ask[i], V_ask[i], P_bid[i], V_bid[i]}_i=1^10"""
    assert isinstance(x, tf.Tensor) and x.shape.as_list()[-1] == 40
    if not self._volume_only and self._max_level == 10: return x
    # Apply max_level
    if self._max_level < 10:
      sizes = [4 * self._max_level, 4 * (10 - self._max_level)]
      x, _ = tf.split(x, num_or_size_splits=sizes, axis=-1)
    # Apply volume only
    if self._volume_only:
      dim = x.shape.as_list()[-1]
      indices = [i for i in range(dim) if i % 2 != 0]
      x = tf.gather(x, indices, axis=-1)
    return x
