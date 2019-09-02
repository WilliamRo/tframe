from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class TrainerConfigs(object):
  """TODO: Somehow merge this class to TrainerHub
  """

  train = Flag.boolean(True, 'Whether this is a training task')
  smart_train = Flag.boolean(False, 'Whether to use smart trainer', is_key=None)
  save_model = Flag.boolean(True, 'Whether to save model during training')
  save_model_at_the_end = Flag.boolean(False, '...')
  overwrite = Flag.boolean(False, 'Whether to overwrite records')
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

  val_preheat = Flag.integer(0, 'metric = metric_batch[val_preheat:].mean')
  val_batch_size = Flag.integer(None, 'Batch size in batch validation')
  eval_batch_size = Flag.integer(None, 'Batch size in batch evaluation')

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
    False, 'Whether to save variables in optimizer for training')

  uncertain_round_len = Flag.boolean(
    False, 'Whether tframe is uncertain about the round length ', is_key=False)

  supreme_reset_flag = Flag.boolean(
    None, 'Reset buffer option of highest priority')

