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
    True, 'If set to True, agent will gather information in a default way'
          ' when export_note flag is set to True')

  note_cycle = Flag.integer(0, 'Note cycle')
  note_per_round = Flag.integer(0, 'Note per round')

  export_structure_detail = Flag.boolean(
    True, 'Option to export structure detail')

  # TODO: ++export_tensors
  export_tensors_to_note = Flag.boolean(
    False, 'Whether to export tensors to note')
  export_tensors_upon_validation = Flag.boolean(
    False, 'Whether to export tensors after validation')
  export_states = Flag.boolean(False, '...')
  export_dy_ds = Flag.boolean(False, '...')
  export_gates = Flag.boolean(False, '...')
  export_weights = Flag.boolean(False, '...')
  export_masked_weights = Flag.boolean(False, '...')
  export_bias = Flag.boolean(False, '...')
  export_weight_grads = Flag.boolean(False, '...')
  export_kernel = Flag.boolean(False, '...')
  use_default_s_in_dy_ds = Flag.boolean(False, '...')
  calculate_mean = Flag.boolean(False, '...')

  export_sparse_weights = Flag.boolean(False, '...')
  export_activations = Flag.boolean(False, '...')

  export_dl_dx = Flag.boolean(False, '...')
  export_dl_ds_stat = Flag.boolean(False, '...')
  export_jacobian_norm = Flag.boolean(False, '...')
  error_injection_step = Flag.integer(-1, '...')
  max_states_per_block = Flag.integer(
    -1, 'Max state size for each state block to export')

  export_top_k = Flag.integer(3, 'Used in export_false of classifier')

  # Statistics only for note summary
  total_params = Flag.integer(0, 'Parameter #', is_key=None)
  dense_total_params = Flag.integer(
    0, 'Parameter # before pruning', is_key=None)
  serial_num = Flag.integer(1, '...', is_key=None)
  supplement = Flag.string(None, 'Supplement', is_key=None)

  # TODO: remove this flag
  take_note_in_beginning = Flag.boolean(
    False, 'Whether to take note on 1st iteration')

  export_input_connection = Flag.boolean(
    False, 'Whether to export input connection stats like a heat map.'
           ' Usually when this option is set to True, a dataset related'
           ' extractor will be registered to net.variable extractor, '
           'as it does in MNIST tasks.')

  gather_only_scalars = Flag.boolean(False, 'Whether to export only scalars')

  export_var_alpha = Flag.boolean(
    False, 'Vars list: (1) coef in sig_curve;')

  take_down_confusion_matrix = Flag.boolean(
    False, 'Whether to take down confusion matrix after evaluation')


  def smooth_out_note_configs(self):
    if self.use_default_s_in_dy_ds: self.export_dy_ds = True

    if (self.export_dy_ds or self.export_gates or self.export_states or
        self.export_weights or self.export_bias or self.export_kernel or
        self.export_dl_dx or self.export_dl_ds_stat or
        self.export_sparse_weights):
      self.export_tensors_to_note = True


