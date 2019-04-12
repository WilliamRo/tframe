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
  summary = Flag.boolean(True, 'Whether to write summary')
  epoch_as_step = Flag.boolean(True, '...')
  snapshot = Flag.boolean(False, 'Whether to take snapshot during training')
  evaluate_model = Flag.boolean(
    False, 'Whether to evaluate model after training')

  val_preheat = Flag.integer(0, 'metric = metric_batch[val_preheat:].mean')
  val_batch_size = Flag.integer(None, 'Batch size in batch validation')

  block_validation = Flag.whatever(False, '???')
  rand_over_classes = Flag.boolean(False, '...', is_key=None)

  sample_num = Flag.integer(9, 'Sample number in some unsupervised learning '
                               'tasks')

  clip_threshold = Flag.float(
    0., 'Threshold for clipping gradients', is_key=None)
  clip_method = Flag.string('norm', 'Gradient clip method', is_key=None)
