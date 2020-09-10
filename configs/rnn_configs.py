from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class RNNConfigs(object):
  val_num_steps = Flag.integer(-1, 'Validation number steps')
  # VIC
  allow_loss_in_loop = Flag.boolean(
    False, 'Set to False to avoid being slow')

  # Training configs
  parallel_on = Flag.boolean(False, 'Whether to turn on parallel option')

  # Basic RNN configs
  rc_dims = Flag.whatever(None, '...')
  notify_when_reset = Flag.whatever(False, '...')
  truncate_grad = Flag.boolean(False, 'Whether to truncate gradient in RNN',
                               is_key=None)
  forward_gate = Flag.boolean(
    None, 'Whether to calculate units using gate units from previous time '
          'step', is_key=None)

  # LSTM-based configs
  state_size = Flag.integer(0, 'State size for some specific RNN cells',
                            is_key=None)
  neurons_per_unit = Flag.integer(3, '...', is_key=None)
  input_gate = Flag.boolean(True, 'Whether to use input gate in LSTM',
                            is_key=None)
  forget_gate = Flag.boolean(True, 'Whether to use forget gate in LSTM',
                             is_key=None)
  output_gate = Flag.boolean(True, 'Whether to use output gate in LSTM',
                             is_key=None)
  input_bias_initializer = Flag.float(
    0., 'input gate bias initializer', is_key=None)
  output_bias_initializer = Flag.float(
    0., 'output gate bias initializer', is_key=None)
  forget_bias_initializer = Flag.float(
    0., 'forget gate bias initializer', is_key=None)
  fc_memory = Flag.boolean(True, 'Whether to fully connect memory', is_key=None)
  act_memory = Flag.boolean(True, 'Whether to activate memory', is_key=None)
  with_peepholes = Flag.boolean(False, 'Whether to add peepholes in LSTM',
                                is_key=None)
  output_activation = Flag.boolean(
    True, 'Whether to use output activation (especially in LSTM)', is_key=None)
  couple_gates = Flag.boolean(
    False, 'Whether to couple alpha and beta gates', is_key=None)

  # AMU configs
  num_units = Flag.integer(0, 'Units# used in AMU model', is_key=None)
  unit_size = Flag.integer(0, 'Size for each AMU', is_key=None)

  # Shem configs
  output_as_mem = Flag.boolean(
    True, 'Whether to pass forward output as cell state', is_key=None)

  # Ham configs
  mem_cfg = Flag.string('', 'e.g. `7-a-f;8-na-nf`')
  short_mem_size = Flag.integer(0, 'Size of short-term memory units',
                                is_key=None)
  short_in_gate = Flag.boolean(
    False, 'Whether to use input gate for short-term memory units',
    is_key=None)
  short_forget_gate = Flag.boolean(
    False, 'Whether to use forget gate for short-term memory units',
    is_key=None)
  short_out_gate = Flag.boolean(
    False, 'Whether to use output gate for short-term memory units',
    is_key=None)
  short_act_mem = Flag.boolean(
    False, 'Whether to activate memory for short-term memory units',
    is_key=None)
  short_fc_mem = Flag.boolean(
    False, 'Whether to fully connect memory for short-term memory units',
    is_key=None)
  long_mem_size = Flag.integer(0, 'Size of long-term memory units', is_key=None)
  long_in_gate = Flag.boolean(
    False, 'Whether to use input gate for long-term memory units', is_key=None)
  long_forget_gate = Flag.boolean(
    False, 'Whether to use forget gate for long-term memory units', is_key=None)
  long_out_gate = Flag.boolean(
    False, 'Whether to use output gate for long-term memory units', is_key=None)
  long_act_mem = Flag.boolean(
    False, 'Whether to activate memory for long-term memory units', is_key=None)
  long_fc_mem = Flag.boolean(
    False, 'Whether to fully connect memory for long-term memory units',
    is_key=None)
  use_mem_wisely = Flag.boolean(
    False, 'Whether to use memory in a wise way', is_key=None)
  train_gates = Flag.boolean(False, 'Whether to train gates', is_key=None)
  apply_default_gate_loss = Flag.boolean(
    True, 'Whether to use indiscriminate gate loss in training', is_key=None)

  use_conveyor = Flag.boolean(
    False, 'Whether to use conveyor for accessing previous inputs', is_key=None)
  conveyor_length = Flag.integer(None, 'Length of conveyor', is_key=None)
  conveyor_input_shape = Flag.list(
    None, 'This is a compromising variable for conveyor logic. To be modified')
  tape_length = Flag.integer(None, 'Length of tape', is_key=None)

  # Clockwork RNN
  periods = Flag.whatever(None, 'Periods for each module in CWRNN', is_key=None)

  # BETA:
  use_rtrl = Flag.boolean(
    False, 'Whether to use RTRL in training RNN', is_key=None)
  test_grad = Flag.boolean(False, '...')

  # GDU configs
  gdu_string = Flag.string(None, 'Config string', is_key=None)
  temporal_reverse = Flag.boolean(
    False, 'Reverse alpha and beta gate temporally', is_key=None)
  spatial_reverse = Flag.boolean(
    False, 'Reverse alpha and beta gate spatially', is_key=None)
  temporal_configs = Flag.string(None, 'Temporal config string', is_key=None)
  temporal_activation = Flag.string('tanh', 'Temporal activation', is_key=None)
  spatial_dim = Flag.integer(None, 'Spatial dimension', is_key=None)
  spatial_configs = Flag.string(None, 'Spatial config string', is_key=None)
  spatial_activation = Flag.string('relu', 'Spatial activation', is_key=None)
  use_reset_gate = Flag.boolean(False, 'Whether to use reset gate in model',
                                is_key=None)
  reset_gate_style = Flag.string(
    's', '`s` or `a`, corresponding to 2 variants of GRU', is_key=None)

  shunt_output = Flag.boolean(False, 'Whether to shunt output', is_key=None)
  gate_output = Flag.boolean(False, 'Whether to gate output', is_key=None)

  psi_string = Flag.string(None, 'Config string for psi', is_key=None)

  group_string = Flag.string(None, 'Group string', is_key=None)

  fast_size = Flag.integer(None, 'Fast cell size in fsrnn', is_key=None)
  fast_layers = Flag.integer(None, 'Fast layer number in fsrnn', is_key=None)
  slow_size = Flag.integer(None, 'Slow cell size in fsrnn', is_key=None)

  cell_zoneout = Flag.float(0.0, 'Cell zoneout for LSTM', is_key=None)
  hidden_zoneout = Flag.float(0.0, 'Hidden zoneout for LSTM', is_key=None)

  val_info_splits = Flag.integer(
    0, 'If is not 0, metric of different part of sequence will be displayed'
       ' during validation in training model handling tasks with long '
       'sequences, e.g. cPTB')

  # GAM-RHN configs
  gam_config = Flag.string(None, 'Gam config string (SxN)', is_key=None)
  sog_version = Flag.integer(
    0, 'Version of `softmax over groups` activation. '
       'v0 bases on reshape, v1 bases on matmul')
  gam_read_version = Flag.integer(
    0, 'Version of read operation from GAM. '
       'v0 bases on reshape, slow but save space.'
       'v1 bases on matmul, quick but needs more space.')
  gam_dropout = Flag.float(0.0, 'Dropout for GAM', is_key=None)
  rhn_dropout = Flag.float(0.0, 'Dropout for RHN', is_key=None)
  sparse_gam = Flag.boolean(True, 'Whether to use sparse tensor in GAM')
  head_bias = Flag.boolean(
    False, 'Whether to use bias in gam head neurons', is_key=None)
  address_bias = Flag.boolean(
    False, 'Whether to use bias in gam address neurons', is_key=None)
  # head_size is defined in model_configs.py

  # Mogrifier
  mog_rounds = Flag.integer(5, 'Update rounds in Mogrifier models', is_key=None)
  mog_lower_rank = Flag.integer(
    None, 'Lower rank used in Mogrifier models', is_key=None)

