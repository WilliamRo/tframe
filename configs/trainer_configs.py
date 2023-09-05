from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf
from .flag import Flag
from tframe.utils.arg_parser import Parser


class TrainerConfigs(object):
  """TODO: Somehow merge this class to TrainerHub
  """

  train = Flag.boolean(True, 'Whether this is a training task')
  smart_train = Flag.boolean(False, 'Whether to use smart trainer', is_key=None)
  save_model = Flag.boolean(True, 'Whether to save model during training')
  save_model_in_the_beginning = Flag.boolean(False, '...')
  save_model_at_the_end = Flag.boolean(False, '...')
  overwrite = Flag.boolean(True, 'Whether to overwrite records')
  summary = Flag.boolean(False, 'Whether to write summary')
  epoch_as_step = Flag.boolean(True, '...')
  snapshot = Flag.boolean(False, 'Whether to take snapshot during training')
  evaluate_model = Flag.boolean(
    False, 'Whether to evaluate model after training')
  evaluate_train_set = Flag.boolean(
    False, 'Whether to evaluate train set after training')
  evaluate_val_set = Flag.boolean(
    False, 'Whether to evaluate validation set after training')
  evaluate_test_set = Flag.boolean(
    False, 'Whether to evaluate test set after training')
  early_stop_metric = Flag.string(None, 'Early-stop metric', is_key=None)
  batch_metric = Flag.string(None, 'Batch metric during training')
  eval_metric = Flag.string(None, 'Metric for evaluation')

  save_records = Flag.boolean(True, 'Whether to save records')
  clear_records_before_training = Flag.boolean(
    False, 'Whether to clear records before training')

  val_preheat = Flag.integer(0, 'metric = metric_batch[val_preheat:].mean')
  val_batch_size = Flag.integer(None, 'Batch size in batch validation')
  eval_batch_size = Flag.integer(None, 'Batch size in batch evaluation')
  eval_num_steps = Flag.integer(None, 'Step number in batch evaluation')

  batchlet_size = Flag.integer(None, 'Batchlet size', is_key=False)
  gradlet_in_device = Flag.boolean(
    True, 'Option to keep gradients in device in batchlet training')

  block_validation = Flag.whatever(False, '???')
  rand_over_classes = Flag.boolean(False, '...', is_key=None)

  sample_num = Flag.integer(9, 'Sample number in some unsupervised learning '
                               'tasks')

  clip_threshold = Flag.float(
    0., 'Threshold for clipping gradients', is_key=None)
  clip_method = Flag.string('norm', 'Gradient clip method', is_key=None)

  terminate_on_note = Flag.boolean(
    False, 'This option is for the convenience of taking notes')

  val_progress_bar = Flag.boolean(
    False, 'Whether to show progress bar during validation')
  val_decimals = Flag.integer(3, 'Decimals displayed in validation reports')

  save_train_opt_vars = Flag.boolean(
    True, 'Whether to save variables in optimizer for training')

  uncertain_round_len = Flag.boolean(
    False, 'Whether tframe is uncertain about the round length ', is_key=False)

  supreme_reset_flag = Flag.boolean(
    None, 'Reset buffer option of highest priority')

  beta1 = Flag.float(0.9, 'beta1 in Adam', is_key=None)
  beta2 = Flag.float(0.999, 'beta2 in Adam', is_key=None)

  use_global_regularizer = Flag.boolean(
    False, 'Whether to use global regularizer', is_key=None)
  global_l1_penalty = Flag.float(0.0, 'Global l1 penalty', is_key=None)
  global_l2_penalty = Flag.float(0.0, 'Global l2 penalty', is_key=None)
  l1_penalty = Flag.float(0.0, 'l1 penalty', is_key=None)
  l2_penalty = Flag.float(0.0, 'l2 penalty', is_key=None)

  decoupled_l2_penalty = Flag.float(
    0.0, 'l2 penalty for decoupled weight decay', is_key=None)

  saturation_penalty = Flag.float(None, 'Saturation penalty', is_key=None)
  encourage_saturation = Flag.boolean(
    True, 'Whether to encourage saturation', is_key=None)

  regularizer = Flag.string('l2', 'Regularizer', name='reg', is_key=None)
  global_constraint = Flag.string(
    None, 'Global constraint. Currently only used in kernel_base', is_key=None)
  reg_strength = Flag.float(0.0, 'Regularizer strength', name='reg_str',
                            is_key=None)

  opt_lr_multiplier = Flag.float(
    1.0, 'Learning rate decay applied via tframe optimizer')
  opt_nan_protection = Flag.boolean(
    False, 'Whether to use NaN protection in tframe optimizer')
  state_nan_protection = Flag.boolean(
    False, 'Whether to use NaN protection on train state update. '
           'Usually used with clip_nan_protection')
  terminate_on_nan = Flag.boolean(True, 'Whether to terminate on NaN')
  lives = Flag.integer(0, 'Number of chances to resurrect', is_key=None)
  reset_optimizer_after_resurrection = Flag.boolean(
    False, 'Whether to re-initiate optimizer after resurrection. '
           'Take effect only when lr_decay < 1')
  optimizer_epsilon = Flag.float(
    1e-8, 'epsilon used for initiating Optimizers', is_key=None)

  lr_decay_method = Flag.string(None, 'Learning rate decay method', is_key=None)

  bs_mar = Flag.float(
    None, 'Batch size modifier after resurrection', is_key=None)

  updates_per_round = Flag.integer(None, 'Number of updates per round')

  wmxe_min = Flag.float(
    0.1, 'Minimum value of weights used in weighted M[SA]E', is_key=None)

  probe_func = Flag.whatever(None, 'Probe function')
  customized_tensors_to_export = Flag.whatever(
    None, 'Customized tensors to export')

  balance_classes = Flag.boolean(
    False, 'Whether to balance classes during training', is_key=None)


  @property
  def lr_decay_enabled(self):
    """If this is true:
       (1) variables for global_step and decay_steps will initiated.
       (2) decay_steps will be set before trainer's outer loop or after
           resurrection
       (3) global_step will be reset to 0 each time a new record appears
       (4) global_step will be increased by 1 after each update   """
    if self.lr_decay_method is None: return False
    return self.lr_decay_method not in ('', '-', 'none', 'None')

  def get_global_regularizer(self):
    """Used only in kernel base"""
    if not self.use_global_regularizer: return None
    from tframe import regularizers
    return regularizers.get(self.regularizer)

  def get_global_constraint(self):
    """Used only in kernel base"""
    if self.global_constraint in [None, '']: return None
    from tframe import constraints
    return constraints.get(self.global_constraint)

  def smooth_out_trainer_configs(self):
    if self.global_l2_penalty * self.decoupled_l2_penalty != 0:
      raise AssertionError(
        '!! global_l2_penalty can not be used with decoupled_l2_penalty')

