from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class ModelConfigs(object):

  mark = Flag.string(None, 'Model identifier', is_key=True)
  prefix = Flag.string(None, 'Prefix to mark')
  suffix = Flag.string(None, 'Suffix to mark')
  model = Flag.whatever(None, 'A function which returns a built model')
  learning_rate = Flag.float(None, 'Learning rate', name='lr', is_key=None)
  regularizer = Flag.string('l2', 'Regularizer', name='reg', is_key=None)
  reg_strength = Flag.float(0.0, 'Regularizer strength', name='reg_str',
                            is_key=None)
  weight_initializer = Flag.whatever(None, 'Weight initializer')
  bias_initializer = Flag.whatever(None, 'Bias initializer')
  actype1 = Flag.string('relu', 'Activation type 1', is_key=None)
  actype2 = Flag.string('relu', 'Activation type 2', is_key=None)
  use_bias = Flag.boolean(True, 'Whether to use bias', is_key=None)
  use_batchnorm = Flag.boolean(False, 'Whether to use batch norm', is_key=None)

  hidden_dim = Flag.integer(-1, 'Hidden dimension', is_key=None)
  fc_dims = Flag.whatever(None, '...')
  num_blocks = Flag.integer(-1, 'Block number in model', is_key=None)
  input_shape = Flag.list([], 'Input shape of samples')
  output_dim = Flag.integer(0, 'Output dimension of a model')
  num_classes = Flag.integer(-1, 'Class number for classification tasks')
  memory_depth = Flag.integer(1, 'Memory depth for system identification')
  loss_function = Flag.whatever('cross_entropy', 'Loss function')
  use_logits = Flag.boolean(False, 'Whether to use logits to calculate losses')
  last_only = Flag.boolean(
    False, 'Whether to use only the value in the last step in sequence '
           'prediction tasks')
  optimizer = Flag.whatever(None, 'optimizer')

  output_size = Flag.integer(0, 'Output dimension for a single layer',
                             is_key=None)
  add_customized_loss = Flag.boolean(False, 'Whether to add customized loss',
                                     is_key=None)
  gate_loss_strength = Flag.float(0.01, 'Strength for gate loss', is_key=None)

  show_extra_loss_info = Flag.boolean(
    False, 'Whether to show extra loss info while predictor calls '
           'net.extra_loss')

