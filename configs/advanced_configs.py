
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

  @property
  def cl_reg_on(self):
    return self.cl_reg_config is not None

  # endregion : Dynamic Evaluation

  # region : Attention Related

  pos_encode = Flag.string(None, 'Type of positional encoding', is_key=None)
  spatial_heads = Flag.integer(None, 'Number of spatial heads', is_key=None)
  temporal_heads = Flag.integer(None, 'Number of temporal heads', is_key=None)

  # endregion : Attention Related

  # region : Etching

  force_to_use_pruner = Flag.boolean(
    False, 'Used in neurobase.py -> dense_rn; model.py; kernel_base.py')
  etch_quietly = Flag.boolean(True, 'Whether to etch quietly')
  max_flip = Flag.float(None, 'Max flip momentum', is_key=None)
  init_flip = Flag.float(None, 'Initial flips', is_key=None)
  flip_beta = Flag.float(1.0, 'Flip momentum decay coef', is_key=None)
  flip_alpha = Flag.float(0.99, 'Flip momentum add coef', is_key=None)
  flip_irreversible = Flag.boolean(
    True, 'Whether to forbid regrowth in flip prune', is_key=None)

  force_initialize = Flag.boolean(
    False, 'This flag is used for examine lottery rewind', is_key=None)
  forbid_lottery_saving = Flag.boolean(
    False, 'This flag is also used for examine lottery rewind', is_key=None)

  # endregion : Etching

  # region: Quantization

  binarize_weights = Flag.boolean(
    False, 'Whether to binarize weights in kernel', is_key=None)
  clip_weight_at = Flag.float(
    None, 'Weight clip during updating, usually used with `binarize_weights`',
    is_key=None)

  # endregion: Quantization

  # region: Incremental Learning

  cl_reg_config = Flag.string(
    None, 'Configuration of continual learning using regularization method',
    is_key=None)
  cl_reg_lambda = Flag.float(0.0, 'lambda for cl reg methods', is_key=None)

  create_shadow_vars = Flag.boolean(
    False, 'Whether to create shadows for all trainable vars')
  save_shadow_vars = Flag.boolean(
    True, 'Whether to save shadow variables after training')

  # endregion: Incremental Learning


