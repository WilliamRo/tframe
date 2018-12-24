from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class RNNConfigs(object):
  # Training configs
  parallel_on = Flag.boolean(False, 'Whether to turn on parallel option')

  # Basic RNN configs
  rc_dims = Flag.whatever(None, '...')
  notify_when_reset = Flag.whatever(False, '...')
  truncate_grad = Flag.boolean(None, 'Whether to truncate gradient in RNN',
                               is_key=None)
  forward_gate = Flag.boolean(
    None, 'Whether to calculate units using gate units from previous time '
          'step', is_key=None)

  # LSTM-based configs
  neurons_per_unit = Flag.integer(3, '...', is_key=None)
  input_gate = Flag.boolean(True, 'Whether to use input gate in LSTM',
                            is_key=None)
  forget_gate = Flag.boolean(True, 'Whether to use forget gate in LSTM',
                             is_key=None)
  output_gate = Flag.boolean(True, 'Whether to use output gate in LSTM',
                             is_key=None)
  fc_memory = Flag.boolean(True, 'Whether to fully connect memory', is_key=None)
  act_memory = Flag.boolean(True, 'Whether to activate memory', is_key=None)
  with_peepholes = Flag.boolean(False, 'Whether to add peepholes in LSTM',
                                is_key=None)

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

  # BETA:
  use_rtrl = Flag.boolean(
    False, 'Whether to use RTRL in training RNN', is_key=None)
  test_grad = Flag.boolean(False, '...')
