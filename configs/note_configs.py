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

  # TODO: ++export_tensor
  export_tensors_to_note = Flag.boolean(
    False, 'Whether to export tensors to note')


  def smooth_out_note_configs(self):
    pass
    # if self.gather_notes: self.export_note = True
