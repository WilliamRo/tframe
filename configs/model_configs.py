from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class ModelConfigs(object):

  mark = Flag.string(None, 'Model identifier', is_key=True)
  prefix = Flag.string(None, 'Prefix to mark')
  suffix = Flag.string(None, 'Suffix to mark')
  script_suffix = Flag.string(None, 'Suffix used in script helper')
  branch_suffix = Flag.string(
    '', 'Suffix added to mark after model has been loaded')
  specified_ckpt_path = Flag.string(
    None, 'Specified checkpoints path used in Agent.load method')
  model = Flag.whatever(None, 'A function which returns a built model')
  archi_string = Flag.string(
    None, 'Architecture string for parsing', is_key=None)
  identifier = Flag.string(
    '', 'Model identifier, used in summary viewer', is_key=None)
  learning_rate = Flag.float(None, 'Learning rate', name='lr', is_key=None)
  momentum = Flag.float(0.9, 'Momentum', is_key=None)
  weight_initializer = Flag.string(None, 'Weight initializer', is_key=None)
  bias_initializer = Flag.float(None, 'Bias initializer', is_key=None)
  actype1 = Flag.string(None, 'Activation type 1', is_key=None)
  actype2 = Flag.string(None, 'Activation type 2', is_key=None)
  use_bias = Flag.boolean(True, 'Whether to use bias', is_key=None)
  use_batchnorm = Flag.boolean(False, 'Whether to use batch norm', is_key=None)

  hidden_dim = Flag.integer(-1, 'Hidden dimension', is_key=None)
  fc_dims = Flag.whatever(None, '...')
  num_blocks = Flag.integer(-1, 'Block number in model', is_key=None)
  num_layers = Flag.integer(1, 'Layer number', is_key=None)
  num_concurrent = Flag.integer(1, 'Concurrent number', is_key=None)
  layer_width = Flag.integer(None, 'Layer width', is_key=None)
  input_shape = Flag.list([], 'Input shape of samples')
  output_dim = Flag.integer(0, 'Output dimension of a model')
  target_dim = Flag.integer(0, 'User specified target dim of a model')
  target_dtype = Flag.whatever(None, 'Target data type')
  num_classes = Flag.integer(-1, 'Class number for classification tasks')
  memory_depth = Flag.integer(1, 'Memory depth for system identification')
  loss_function = Flag.whatever('cross_entropy', 'Loss function')
  loss_string = Flag.string(None, 'Loss string', is_key=None)
  use_logits = Flag.boolean(
    False, 'Whether to use logits to calculate losses', is_key=None)
  last_only = Flag.boolean(
    False, 'Whether to use only the value in the last step in sequence '
           'prediction tasks', is_key=None)
  optimizer = Flag.whatever(None, 'optimizer', is_key=None)

  output_size = Flag.integer(0, 'Output dimension for a single layer',
                             is_key=None)
  bias_out_units = Flag.boolean(True, 'Whether to bias output units',
                                is_key=None)
  add_customized_loss = Flag.boolean(False, 'Whether to add customized loss',
                                     is_key=None)
  gate_loss_strength = Flag.float(0.01, 'Strength for gate loss', is_key=None)

  show_extra_loss_info = Flag.boolean(
    False, 'Whether to show extra loss info while predictor calls '
           'net.extra_loss')

  dropout = Flag.float(0.0, 'Dropout rate', is_key=None)
  input_dropout = Flag.float(0.0, 'Dropout rate', is_key=None)
  output_dropout = Flag.float(0.0, 'Dropout rate', is_key=None)
  forward_dropout = Flag.boolean(
    False, 'Whether to use forward dropout', is_key=None)

  use_recurrent_net = Flag.boolean(False, 'Whether to use recurrent net')

  use_bit_max = Flag.boolean(False, 'Whether to use bitmax', is_key=None)
  use_softmax = Flag.boolean(False, 'Whether to use softmax', is_key=None)
  num_heads = Flag.integer(1, 'Head #', is_key=None)

  centralize_data = Flag.boolean(
    False, 'Whether to centralize data', is_key=None)
  data_mean = Flag.float(None, 'Used for normalizing data set')
  data_std = Flag.float(None, 'Used for normalizing data set')

  etch_on = Flag.boolean(False, 'Whether to activate weights etching')
  prune_on = Flag.boolean(False, 'Whether lottery option is activated. '
                                 'Should only be set in smooth_out ...')
  pruning_rate_fc = Flag.float(
    0.0, 'Pruning rate for fully connected layers', is_key=None)
  pruning_iterations = Flag.integer(0, 'Pruning iterations', is_key=None)
  weights_fraction = Flag.float(None, 'Master weights fraction', is_key=None)
  weights_mask_on = Flag.boolean(False, 'Whether to use pruner')

  head_size = Flag.integer(None, 'Head size of a HHD', is_key=None)
  sigmoid_coef = Flag.float(
    1.0, 'Used in narrow the effective domain of sigmoid function', is_key=None)
  full_weight = Flag.boolean(
    False, 'Whether to use full weight matrix in sparse method', is_key=None)
  mask_option = Flag.string(None, 'Mask option', is_key=None)
  factoring_dim = Flag.integer(
    0, 'Factoring dimention, ref: fcrbm09', is_key=None)
  layer_normalization = Flag.boolean(False, 'Whether to use LN', is_key=None)
  normalize_each_psi = Flag.boolean(
    False, 'Wheter to normalize each psi during LN', is_key=None)
  gain_initializer = Flag.float(1.0, 'Gain initializer for LN', is_key=None)
  variance_epsilon = Flag.float(
    1e-6, 'A small float number to avoid dividing by 0')

  hyper_kernel = Flag.string(None, 'Kernel used in hyper model', is_key=None)
  hyper_dim = Flag.integer(None, 'Dimension of hyper seed', is_key=None)
  signal_size = Flag.integer(None, 'Hyper signal size', is_key=None)

  etch_string = Flag.string(None, 'An etch config string', is_key=None)

  lottery_kernel = Flag.string(
    'lottery18', 'Method used in Lottery(EtchKernel) to get new mask',
    is_key=None)
  use_gate_as_mask = Flag.boolean(False, 'Whether to use gates as masks',
                                  is_key=None, name='gam')
  rec_dropout = Flag.float(0.0, 'Recurrent dropout', is_key=None)
  zoneout = Flag.float(0.0, 'Zoneout rate', is_key=None)

  weight_dropout = Flag.float(None, 'Weight dropout', is_key=None)
  global_weight_dropout = Flag.float(0.0, 'Global weight dropout', is_key=None)

  use_gather_indices = Flag.boolean(
    True, 'Option to enable training irregular sequences in batches')

  gutter = Flag.boolean(False, 'Whether to use gutter', is_key=None)
  gutter_bias = Flag.float(None, 'Gutter bias', is_key=None)
  early_stop_on_loss = Flag.boolean(
    False, 'Whether to use loss as early stop criterion', is_key=True)
  suppress_current_graph = Flag.boolean(
    False, 'Option to suppress current graph')

  use_lob_sig_curve = Flag.boolean(
    False, 'Whether to use significance curve for model in tasks on LOB',
    is_key=None)
  lob_fix_sig_curve = Flag.boolean(
    False, 'Whether to fix significant curve', is_key=None)

  max_norm = Flag.float(None, 'Max norm constraint on variables', is_key=None)


  def smooth_out_model_configs(self):
    if self.pruning_rate_fc > 0: self.prune_on = True
    if self.prune_on and self.etch_on:
      raise AssertionError('!! prune and etch can not be on in the same time')
