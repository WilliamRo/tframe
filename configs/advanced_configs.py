
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class AdvancedConfigs(object):

  # region : Dynamic Evaluation

  dynamic_evaluation = Flag.boolean(
    False, 'Whether to turn on dynamic evaluation', is_key=None)
  train_stats_exists = Flag.boolean(
    False, 'Whether or not train_stats used in dynamic evaluation exists')
  de_save_train_stats = Flag.boolean(False, 'Whether to save train stats')
  de_val_size = Flag.integer(
    -1, 'Size of validation set used in dynamic evaluation')
  de_val_pct = Flag.float(
    1.0, 'Proportion of validation set used in dynamic evaluation')
  de_num_steps = Flag.integer(
    20, 'According to Krause2018, this value should be 20 for c-level model '
        'and 5 for w-level model', is_key=None)
  de_batch_size = Flag.integer(
    100, 'Batch size used in calculating gradient stats. Ref: Krause2018',
    is_key=None)
  de_max_batches = Flag.integer(
    -1, 'Max batches used in evaluating gradstats')
  de_eta = Flag.string(
    None, 'Learning rate used in krause evaluator, Can be a list separated'
          ' by comma to provide a search list')
  de_lambda = Flag.string(
    None, 'Decay rate used in krause evaluator. Can be a list separated by'
          ' comma to provide a search list.')
  de_eval_val_set = Flag.boolean(
    False, 'Whether to evaluate validation set before HP searching')
  de_eval_test_set = Flag.boolean(
    False, 'Whether to evaluate test set in a common way before dynamic '
           'evaluation')
  de_delay = Flag.integer(1, 'Update delay. First used in LOB prediction')

  @property
  def de_eta_option(self):
    if self.de_eta is None: return None
    return Flag.parse_comma(self.de_eta, float)

  @property
  def de_lambda_option(self):
    if self.de_lambda is None: return None
    return Flag.parse_comma(self.de_lambda, float)

  # endregion : Dynamic Evaluation


