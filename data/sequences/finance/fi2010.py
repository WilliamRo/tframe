from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tframe import tf

import numpy as np
from tframe import console

from tframe.data.dataset import DataSet
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.base_classes import DataAgent

from tframe.utils.display.progress_bar import ProgressBar
from tframe.utils.display.table import Table
from tframe.utils.janitor import recover_seq_set_outputs
import tframe.utils.maths.wise_man as wise_man

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
  For LOBs with NoAuction
  Day       1      2      3      4      5      6      7      8      9     10
  Blocks  39512  38397  28535  37023  34785  39152  37346  55478  52172  31937

  Reference:
  [1] Adamantios Ntakaris, Martin Magris, Juho Kanniainen, Moncef Gabbouj
      and Alexandros Iosifidis. Benchmark Dataset for Mid-Price Forecasting of
      Limit Order Book Data with Machine Learning Methods. 2018.
  """

  DATA_NAME = 'FI-2010'
  DATA_URL = 'https://etsin.fairdata.fi/api/dl?cr_id=73eb48d7-4dbc-4a10-a52a-da745b47a649&file_id=5b32ac028ab4d130110888f19872320'
  DAY_LENGTH = {
    True:
      [47342, 45114, 33720, 43252, 41171, 47253, 45099, 59973, 57951, 37250],
    False:
      [39512, 38397, 28535, 37023, 34785, 39152, 37346, 55478, 52172, 31937]}
  STOCK_IDs = ['KESBV', 'OUT1V', 'SAMPO', 'RTRKS', 'WRT1V']
  LEN_PER_DAY_PER_STOCK = 'LEN_PER_DAY_PER_STOCK'

  @classmethod
  def load(cls, data_dir, auction=False, norm_type='zscore', setup=2,
           val_size=None, horizon=100, **kwargs):
    should_apply_norm = any(['use_log' not in th.developer_code,
                             'force_norm' in th.developer_code])
    # Sanity check
    assert setup in [1, 2]
    # Load raw LOB data
    lob_set = cls.load_raw_LOBs(data_dir, auction=auction)
    lob_set = cls._init_features_and_targets(lob_set, horizon)
    # Apply setup and normalization
    train_set, test_set = cls._apply_setup(lob_set, setup)
    if should_apply_norm:
      train_set, test_set = cls._apply_normalization(
        train_set, test_set, norm_type)
    if kwargs.get('validate_setup2') and setup == 2 and norm_type == 'zscore':
      cls._validate_setup2(data_dir, auction, train_set)
    return  train_set, test_set

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
      cls, data_dir, auction=False, norm_type='zscore', setup=None,
      file_slices=None, **kwargs):
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
      data_dir, auction, norm_type, type_id, file_slices=file_slices)

    # Wrap raw data into tframe Sequence set
    data_dict = {'raw_data': features}
    data_dict.update(targets)
    seq_set = SequenceSet(data_dict=data_dict, name=cls.DATA_NAME)
    # Save Sequence set
    seq_set.save(data_path)
    console.show_status('Sequence set saved to `{}`'.format(data_path))
    # Return
    return seq_set

  @classmethod
  def divide(cls, lob_set, k_list, first_name, second_name):
    assert isinstance(lob_set, SequenceSet) and lob_set.size <= 5
    if isinstance(k_list, int): k_list = [k_list] * lob_set.size
    first_features, second_features = [], []
    first_targets, second_targets = [], []
    # Separate each stock
    len_per_day_per_stock = lob_set[cls.LEN_PER_DAY_PER_STOCK]
    assert len(len_per_day_per_stock) == lob_set.size
    for stock, (k, lob, move) in enumerate(
        zip(k_list, lob_set.features, lob_set.targets)):
      lengths = len_per_day_per_stock[stock]
      L = sum(lengths[:k])
      if k != 0:
        first_features.append(lob[:L])
        first_targets.append(move[:L])
      if k != len(lengths):
        second_features.append(lob[L:])
        second_targets.append(move[L:])
    # Wrap data sets and return
    first_properties = {
      cls.LEN_PER_DAY_PER_STOCK: [
        s[:k] for k, s in zip(k_list, len_per_day_per_stock) if k != 0]}
    first_set = SequenceSet(
      first_features, first_targets, name=first_name, **first_properties)
    second_properties = {
      cls.LEN_PER_DAY_PER_STOCK: [
        s[k:] for k, s in zip(k_list, len_per_day_per_stock) if k != len(s)]}
    second_set = SequenceSet(
      second_features, second_targets, name=second_name, **second_properties)
    for seq_set in [first_set, second_set]:
      assert np.sum(seq_set.structure) == np.sum(
        np.concatenate(seq_set[cls.LEN_PER_DAY_PER_STOCK]))
    return first_set, second_set

  @classmethod
  def load_raw_LOBs(cls, data_dir, auction=False):
    # Load directly if dataset exists
    data_path = cls._get_data_path(data_dir, auction=auction)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # Otherwise restore raw LOBs from decimal precision data
    dp_set = cls.load_as_tframe_data(
      data_dir, auction=auction, norm_type='decpre', setup=9,
      file_slices=(slice(8, 9), slice(8, 9)))
    # Extract first 40 dimensions in de_set.raw_data
    dp_lob_list = [array[:, :40] for array in dp_set.data_dict['raw_data']]
    # Set parameters for restoration
    p_coef, v_coef = 10000, 100000
    coefs = np.array([p_coef, v_coef] * 20).reshape(1, 40)
    lob_list = [array * coefs for array in dp_lob_list]
    # Check targets
    cls._check_targets(data_dir, auction, dp_set.data_dict)
    # Check lob list
    cls._check_raw_lob(data_dir, auction, lob_list, raise_err=True)

    # Separate sequences for each stock
    # i  0 1 2 3 4 5 6 7
    # --------------------
    #    1 1 0 0 0 1 1 1        := x
    #      1 1 0 0 0 1 1 1
    # d  x 0 1 0 0 1 0 0 x      x[0:2], x[2:5], x[5:8]
    # --------------------
    # j    0 1 2 3 4 5 6
    #        *     *
    # |x[1:] - x[:-1]| reveals cliffs
    LOBs = [[] for _ in range(5)]
    horizons = [10, 20, 30, 50, 100]
    targets = {h: [[] for _ in range(5)] for h in horizons}
    for j, lobs in enumerate(lob_list):
      # Find cliff indices
      max_delta = 300 if auction else 200
      indices = cls._get_cliff_indices(lobs, auction, max_delta=max_delta)
      # Fill LOBs
      from_i = 0
      for stock in range(5):
        to_i = (indices[stock] + 1) if stock < 4 else len(lobs)
        slc = slice(from_i, to_i)
        LOBs[stock].append(lobs[slc])
        for h in horizons: targets[h][stock].append(dp_set.data_dict[h][j][slc])
        if stock != 4: from_i = indices[stock] + 1
    # Generate new data_dict
    data_dict = {h: [np.concatenate(tgt_list) for tgt_list in tgt_lists]
                 for h, tgt_lists in targets.items()}
    data_dict['raw_data'] = [np.concatenate(lb_list) for lb_list in LOBs]
    # Initiate a new seq_set
    seq_set = SequenceSet(
      data_dict=data_dict, name='FI-2010-LOBs',
      **{cls.LEN_PER_DAY_PER_STOCK: cls._get_len_per_day_per_stock(
        data_dir, auction)})
    # Sanity check (394337)
    assert sum(seq_set.structure) == sum(cls.DAY_LENGTH[auction])
    # Save and return
    seq_set.save(filename=data_path)
    console.show_status('{} saved to `{}`'.format(seq_set.name, data_path))
    return seq_set

  # region : Private Methods

  @classmethod
  def _validate_setup2(cls, data_dir, auction, train_set):
    console.show_status('Validating train set ...', '[Setup2-ZScore]')
    assert isinstance(train_set, SequenceSet)
    # Load zscore data set
    zs_set = cls.load_as_tframe_data(
      data_dir, auction=auction, norm_type='zscore', setup=2,
      file_slices=(slice(6, 7), slice(7, 9)))
    zs_feature = zs_set.data_dict['raw_data'][0][:, :40]
    feature = np.concatenate(train_set.features, axis=0)
    assert len(zs_feature) == len(feature)
    delta = np.abs(zs_feature - feature)
    assert np.max(delta) < 1e-4
    console.show_info('Validation completed.')

  @classmethod
  def _init_features_and_targets(cls, lob_set, horizon):
    """x = {P_ask[i], V_ask[i], P_bid[i], V_bid[i]}_i=1^10"""
    max_level = checker.check_positive_integer(th.max_level)
    assert 0 < max_level <= 10
    # Initialize features
    features = lob_set.data_dict['raw_data']
    # .. max_level
    features = [array[:, :4*max_level] for array in features]
    # .. check developer code
    if 'use_log' in th.developer_code:
      for x in features: x[:, 1::2] = np.log10(x[:, 1::2] + 1.0)
      console.show_info('log10 applied to features', '++')
    # .. volume only
    if th.volume_only: features = [array[:, 1::2] for array in features]
    # Set features back
    lob_set.features = features
    # Initialize targets
    lob_set.targets = lob_set.data_dict[horizon]
    return lob_set

  @classmethod
  def _apply_setup(cls, lob_set, setup):
    # Sanity check
    # Currently only Setup2 is supported
    assert setup == 2
    assert isinstance(lob_set, SequenceSet)
    return cls.divide(lob_set, 7, 'Train Set', 'Test Set')

  @classmethod
  def _apply_normalization(cls, train_set, test_set, norm_type):
    assert norm_type == 'zscore'
    assert isinstance(train_set, SequenceSet)
    assert isinstance(test_set, SequenceSet)
    train_x = train_set.stack.features
    mu, sigma = np.mean(train_x, axis=0), np.std(train_x, axis=0)
    train_set.normalize_feature(mu, sigma, element_wise=False)
    test_set.normalize_feature(mu, sigma, element_wise=False)
    return train_set, test_set

  @classmethod
  def _get_cliff_indices(cls, lobs, auction, max_delta=200.0):
    p = lobs[:, 0]
    shift = 1
    delta = np.abs(p[shift:] - p[:-shift])
    indices = np.where(delta > max_delta)[0] + shift - 1
    if auction: indices = [
      i for i in indices
      if np.abs(p[min(i + 100, len(p) - 1)] - p[i - 100]) > max_delta]
    if len(indices) != 4:
      raise AssertionError
    return list(indices)

  @classmethod
  def _get_len_per_day_per_stock(cls, data_dir, auction):
    zscore_set = cls.load_as_tframe_data(
      data_dir, auction=auction, norm_type='zscore', setup=109,
      file_slices=(slice(0, 1), slice(0, 9)))
    lobs_list = zscore_set.data_dict['raw_data']
    assert len(lobs_list) == 10
    lengths = [[] for _ in range(5)]
    for lobs in lobs_list:
      max_delta = 0.4 if auction else 0.1
      indices = cls._get_cliff_indices(lobs, auction, max_delta)
      indices = [-1] + indices + [len(lobs) - 1]
      for i, L in enumerate([indices[j + 1] - indices[j] for j in range(5)]):
        lengths[i].append(L)
    # Sanity check
    assert np.sum(lengths) == sum(cls.DAY_LENGTH[auction])
    return lengths

  @classmethod
  def _load_raw_data(
      cls, data_dir, auction, norm_type, type_id, file_slices=None):
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
    return cls._read_train_test(
      training_set_path, test_set_path, auction, norm_type, file_slices)

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
  def _read_train_test(
      cls, train_dir, test_dir, auction, norm_type, file_slices=None):
    """This method is better used for reading DecPre data for further restoring
    """
    train_paths = cls._get_data_file_path_list(
      train_dir, True, auction, norm_type)
    test_paths = cls._get_data_file_path_list(
      test_dir, False, auction, norm_type)
    # Read data from .txt files
    features, targets = [], {}
    horizons = [10, 20, 30, 50, 100]
    for h in horizons: targets[h] = []
    dim = 144
    if file_slices is None:train_slice, test_slice = slice(0, 1), slice(0, 9)
    else:
      checker.check_type(file_slices, slice)
      assert len(file_slices) == 2
      train_slice, test_slice = file_slices
    data_paths = train_paths[train_slice] + test_paths[test_slice]
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
  def _check_targets(cls, data_dir, auction, data_dict):
    console.show_status('Checking targets list ...')
    assert isinstance(data_dict, dict)
    # Load z-score data
    zscore_set = cls.load_as_tframe_data(
      data_dir, auction=auction, norm_type='zscore', setup=9,
      file_slices=(slice(8, 9), slice(8, 9)))
    assert isinstance(zscore_set, SequenceSet)
    # Check targets
    horizons = [10, 20, 30, 50, 100]
    for h in horizons:
      lob_targets = np.concatenate(data_dict[h])
      zs_targets = np.concatenate(zscore_set.data_dict[h])
      if not np.equal(lob_targets, zs_targets).all():
        raise AssertionError('Targets not equal when horizon = {}'.format(h))
    console.show_info('Targets are all correct.')

  @classmethod
  def _check_raw_lob(cls, data_dir, auction, lob_list, raise_err=False):
    console.show_status('Checking LOB list ...')
    # Sanity check
    assert isinstance(auction, bool) and len(lob_list) == 2
    for lob in lob_list:
      assert isinstance(lob, np.ndarray) and lob.shape[1] == 40
    # Calculate stats for normalization
    lob_1_9 = lob_list[0]
    mu, sigma = np.mean(lob_1_9, axis=0), np.std(lob_1_9, axis=0)
    x_min, x_max = np.min(lob_1_9, axis=0), np.max(lob_1_9, axis=0)
    x_deno = x_max - x_min
    # Load z-score data
    zscore_set = cls.load_as_tframe_data(
      data_dir, auction=auction, norm_type='zscore', setup=9,
      file_slices=(slice(8, 9), slice(8, 9)))
    assert isinstance(zscore_set, SequenceSet)
    zs_all = np.concatenate([
      array[ :, :40] for array in zscore_set.data_dict['raw_data']], axis=0)
    # Load min-max data
    mm_set = cls.load_as_tframe_data(
      data_dir, auction=False, norm_type='minmax', setup=9,
      file_slices=(slice(8, 9), slice(8, 9)))
    mm_all = np.concatenate([
      array[:, :40] for array in mm_set.data_dict['raw_data']], axis=0)
    # Generate lob -> zscore data for validation
    lob_all = np.concatenate(lob_list, axis=0)
    lob_zs_all = (lob_all - mu) / sigma
    # Check error
    max_err = 1e-4
    delta_all = np.abs(lob_zs_all - zs_all)
    if np.max(delta_all) < max_err:
      console.show_info('LOB list is correct.')
      return True
    if raise_err: raise AssertionError
    # Correct LOB using
    console.show_status('Correcting LOB list ...')
    V_errs, P_errs = 0, 0
    bar = ProgressBar(total=len(lob_all))
    for i, j in np.argwhere(delta_all > max_err):
      price_err = j % 2 == 0
      V_errs, P_errs = V_errs + 1 - price_err, P_errs + price_err
      # Find correct value
      val_zs = zs_all[i][j] * sigma[j] + mu[j]
      val_mm = mm_all[i][j] * x_deno[j] + x_min[j]
      zs_mm_err = abs(val_zs - val_mm)
      if zs_mm_err > 0.1:
        raise AssertionError(
          'In LOB[{}, {}] val_zs = {} while val_mm = {}'.format(
            i, j, val_zs, val_mm))
      correct_val = val_mm
      if not P_errs:
        correct_val = np.round(val_mm)
        cor_mm_err = abs(correct_val - val_mm)
        if cor_mm_err > 1e-3:
          raise AssertionError(
            'In LOB[{}, {}] cor_val = {} while val_mm = {}'.format(
              i, j, cor_mm_err, val_mm))
      # Correct value in lob_all
      lob_all[i, j] = correct_val
      bar.show(i)
    # Show status after correction
    console.show_status(
      '{} price errors and {} volume errors have been corrected'.format(
        P_errs, V_errs))
    new_lob_list = []
    for s in [len(array) for array in lob_list]:
      day_block, lob_all = np.split(lob_all, [s])
      new_lob_list.append(day_block)
    assert cls._check_raw_lob(data_dir, auction, new_lob_list, True)
    # for i in range(10): lob_list[i] = new_lob_list[i] TODO
    assert False

  @classmethod
  def _get_data_path(cls, data_dir, auction, norm_type=None, setup=None):
    assert isinstance(auction, bool)
    if all([norm_type is None, setup is None]):
      file_name = 'FI-2010-{}Auction-LOBs.tfds'.format(
        '' if auction else 'No')
    else: file_name = 'FI-2010-{}Auction-{}-Setup{}.tfds'.format(
      '' if auction else 'No', norm_type, setup)
    return os.path.join(data_dir, file_name)

  # endregion : Private Methods

  # region : Public APIs

  @classmethod
  def set_input_shape(cls):
    th.input_shape = [th.max_level * (2 if th.volume_only else 4)]

  # endregion : Public APIs

  # region : RNN batch generator for Sequence Set

  @staticmethod
  def rnn_batch_generator(
      data_set, batch_size, num_steps, is_training, round_len):
    """Generated epoch batches are guaranteed to cover all sequences"""
    assert isinstance(data_set, SequenceSet) and is_training
    L = int(sum(data_set.structure) / batch_size)
    assert L < min(data_set.structure) and L == th.sub_seq_len
    rad = int(th.random_shift_pct * L)
    # Distribute batch_size to stocks
    # [23336, 44874, 38549, 54675, 93316]
    num_sequences = wise_man.apportion(data_set.structure, batch_size)
    # Generate feature list and target list
    features, targets = [], []
    for num, x, y in zip(num_sequences, data_set.features, data_set.targets):
      # Find starts for each sequence to sample
      starts = wise_man.spread(len(x), num, L, rad)
      # Sanity check
      assert len(starts) == num
      # Put the sub-sequences into corresponding lists
      for s in starts:
        features.append(x[s:s+L])
        targets.append(y[s:s+L])
    # Stack features and targets
    features, targets = np.stack(features), np.stack(targets)
    data_set = DataSet(features, targets, is_rnn_input=True)
    assert data_set.size == batch_size
    # Generate RNN batches using DataSet.gen_rnn_batches
    counter = 0
    for batch in data_set.gen_rnn_batches(
        batch_size, num_steps, is_training=True):
      yield batch
      counter += 1

    # Check round_len
    if counter != round_len:
      raise AssertionError(
        '!! counter = {} while round_len = {}. (batch_size = {}, num_steps={})'
        ''.format(counter, round_len, batch_size, num_steps))


  # endregion : RNN batch generator for Sequence Set

  # region : Probe and evaluate

  @staticmethod
  def probe(dataset, trainer):
    from tframe.trainers.trainer import Trainer
    # Sanity check
    assert isinstance(trainer, Trainer) and isinstance(dataset, SequenceSet)
    model = trainer.model
    assert isinstance(model, Classifier)

    # Get table and F1 score for each stock
    label_pred = FI2010._get_label_pred(model, dataset)
    F1s = []
    for lp in label_pred:
      _, F1 = FI2010._get_table_and_F1(lp)
      F1s.append(F1)

    # Get overall table and score
    label_pred = np.concatenate(label_pred)
    table, F1 = FI2010._get_table_and_F1(label_pred, title='All Stocks, ')

    content = 'F1 Scores: {}'.format(', '.join(
      ['[{}] {:.2f}'.format(i + 1, score) for i, score in enumerate(F1s)]))
    return content + table.content

  @staticmethod
  def evaluate(entity, seq_set):
    is_training = isinstance(entity, Trainer)
    if is_training: model = entity.model
    else: model = entity
    # Sanity check
    assert isinstance(model, Classifier) and isinstance(seq_set, SequenceSet)
    # Get table and F1 score for each stock
    label_pred = FI2010._get_label_pred(model, seq_set)
    for i, lp in enumerate(label_pred):
      table, _ = FI2010._get_table_and_F1(lp, title='[{}] '.format(i + 1))
      table.print_buffer()
      if is_training: model.agent.take_notes(table.content)

    # Get overall table and score
    label_pred = np.concatenate(label_pred)
    table, F1 = FI2010._get_table_and_F1(label_pred, title='All Stocks, ')
    table.print_buffer()
    # Take note if is training
    if is_training:
      model.agent.take_notes(table.content)
      model.agent.put_down_criterion('F1 Score', F1)

  @classmethod
  def _get_label_pred(cls, model, dataset):
    # Sanity check
    assert isinstance(model, Classifier)
    # Get predictions and labels
    label_pred_tensor = model.key_metric.quantity_definition.quantities
    label_pred = model.evaluate(label_pred_tensor, dataset, verbose=True)
    console.show_status('Evaluation completed')
    label_pred = recover_seq_set_outputs(label_pred, dataset)
    return label_pred

  @classmethod
  def _get_stats(cls, model, dataset, batch_size=1, report_each_stock=False):
    # Sanity check
    assert isinstance(model, Classifier)
    # Get predictions and labels
    label_pred_tensor = model.key_metric.quantity_definition.quantities
    label_pred = model.evaluate(
      label_pred_tensor, dataset, batch_size=batch_size, verbose=True)
    console.show_status('Evaluation completed')
    table, F1 = cls._get_table_and_F1(label_pred)
    return table, F1

  @classmethod
  def _get_table_and_F1(cls, label_pred, title=''):
    assert isinstance(label_pred, np.ndarray) and label_pred.shape[-1] == 2
    # Initialize table
    movements = ['Upward', 'Stationary', 'Downward']
    header = ['Movement  ', 'Accuracy %', 'Precision %', 'Recall %',
              '   F1 %']
    widths = [len(h) for h in header]
    table = Table(*widths, tab=3, margin=1, buffered=True)
    table.specify_format(*['{:.2f}' for _ in header], align='lrrrr')
    table.hdash()
    table.print('{}Prediction Horizon k = {}'.format(title, th.horizon))
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

  # region : Deprecated

  @classmethod
  def _read_10_days_dep(cls, train_dir, test_dir, auction, norm_type, setup):
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

  # endregion : Deprecated


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
