from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class NoteConfigs(object):
  note_folder_name = Flag.string('notes', '...')

  gather_file_name = Flag.string('gather.txt', '...')
  gather_summ_name = Flag.string('gather.sum', '...')
  show_record_history_in_note = Flag.boolean(False, '...')

  export_note = Flag.boolean(False, 'Whether to take notes')
  gather_note = Flag.boolean(
    False, 'If set to True, agent will gather information in a default way'
          ' when export_note flag is set to True')

  note_cycle = Flag.integer(0, 'Note cycle')
  note_per_round = Flag.integer(0, 'Note per round')

  # TODO: ++export_tensors
  export_tensors_to_note = Flag.boolean(
    False, 'Whether to export tensors to note', is_key=None)
  export_tensors_upon_validation = Flag.boolean(
    False, 'Whether to export tensors after validation')
  export_states = Flag.boolean(False, '...')
  export_dy_ds = Flag.boolean(False, '...')
  export_gates = Flag.boolean(False, '...')
  export_weights = Flag.boolean(False, '...')
  export_masked_weights = Flag.boolean(False, '...')
  export_bias = Flag.boolean(False, '...')
  export_kernel = Flag.boolean(False, '...')
  use_default_s_in_dy_ds = Flag.boolean(False, '...')
  calculate_mean = Flag.boolean(False, '...')

  export_dl_dx = Flag.boolean(False, '...')
  export_dl_ds_stat = Flag.boolean(False, '...')
  export_jacobian_norm = Flag.boolean(False, '...')
  error_injection_step = Flag.integer(-1, '...')
  max_states_per_block = Flag.integer(
    -1, 'Max state size for each state block to export')

  export_top_k = Flag.integer(0, 'Used in export_false of classifier')

  # Statistics only for note summary
  total_params = Flag.integer(0, 'Parameters #', is_key=None)
  serial_num = Flag.integer(1, '...', is_key=None)


  def smooth_out_note_configs(self):
    if self.use_default_s_in_dy_ds: self.export_dy_ds = True

    if (self.export_dy_ds or self.export_gates or self.export_states or
        self.export_weights or self.export_bias or self.export_kernel or
        self.export_dl_dx or self.export_dl_ds_stat):
      self.export_tensors_to_note = True


