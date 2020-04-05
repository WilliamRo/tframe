from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import tkinter as tk
  from tkinter.ttk import Frame

  from PIL import Image as Image_
  from PIL import ImageTk

  from tframe.utils import janitor
  from tframe.utils.note import Note
  from tframe.utils.tensor_viewer import key_events
  from tframe.utils.tensor_viewer.context import Context
  from tframe.utils.tensor_viewer.criteria_figure import CriteriaFigure
  from tframe.utils.tensor_viewer.variable_viewer import VariableViewer

  from tframe.utils.viewer_base.main_frame import Viewer
except Exception as e:
  print(' ! {}'.format(e))
  print(' ! TensorViewer is disabled, install pillow and tkinter to enable it')


class TensorViewer(Viewer):
  """Note Viewer for tframe NOTE"""
  SIZE = 600

  def __init__(self, note=None, note_path=None, init_dir=None, **kwargs):
    # Call parent's initializer
    Viewer.__init__(self)
    self.master.resizable(False, False)
    # Attributes
    self.context = Context()
    self.init_dir = init_dir

    # Define layout
    self.criteria_figure = None
    self.variable_viewer = None
    # Bind keys
    self._bind_key_events()

    # Define layout and refresh
    self._define_layout(**kwargs)
    self._global_refresh()

    # Set plugin (beta) (This line should be put before set_note)
    self._plugins = janitor.wrap(kwargs.get('plugins', []))

    # If note or note_path is provided, try to load it
    if note is not None or note_path is not None:
      self.set_note(note, note_path)

  # region : Public Methods

  def set_note(self, note=None, note_path=None):
    # Set context
    self.context.set_note(note, note_path)

    # Set loss and variables
    assert isinstance(self.criteria_figure, CriteriaFigure)
    ax1_text = 'Step'
    if 'Total Rounds' in note.criteria.keys():
      ax1_text = 'Epoch'
    elif 'Total Iterations' in note.criteria.keys():
      ax1_text = 'Iterations'
    self.criteria_figure.set_context(
      self.context.note.step_array, self.context.note.scalar_dict, ax1_text)

    assert isinstance(self.variable_viewer, VariableViewer)
    tensor_dict = self.context.note.tensor_dict
    if len(tensor_dict) > 0:
      self.variable_viewer.set_variable_dict(
        self.context.note.tensor_dict, self._plugins)
      self.variable_viewer.is_on = True
    else:
      self.variable_viewer.pack_forget()
      self.variable_viewer.is_on = False

    # Global refresh
    self._global_refresh()

    # TODO: somehow necessary
    self.criteria_figure.refresh()

  # endregion : Public Methods

  # region : Private Methods

  def _bind_key_events(self):
    # Bind Key Events
    self.form.bind('<Key>', lambda e: key_events.on_key_press(self, e))
    self.form.bind('<Control-o>', lambda e: key_events.load_note(self, e))

  def _define_layout(self, **kwargs):
    #
    CriteriaFigure.WIDTH = self.SIZE
    CriteriaFigure.HEIGHT = self.SIZE
    self.criteria_figure = CriteriaFigure(self, **kwargs)
    self.criteria_figure.pack(fill=tk.BOTH, side=tk.LEFT)

    #
    VariableViewer.WIDTH = self.SIZE
    VariableViewer.HEIGHT = self.SIZE
    self.variable_viewer = VariableViewer(self)
    self.variable_viewer.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)

    # Relate loss figure and variable viewer
    self.criteria_figure.related_variable_viewer = self.variable_viewer
    self.variable_viewer.related_criteria_figure = self.criteria_figure

    # Pack self
    self.pack(fill=tk.BOTH, expand=True)

  def _global_refresh(self):
    # Refresh title
    title = 'Tensor Viewer'
    if self.context.note_file_name is not None:
      title += ' - {}'.format(self.context.note_file_name)
    else:
      note = self.context.note
      if note: title += ' - {}'.format(note.configs['mark'])
    self.form.title(title)

  # endregion : Private Methods


if __name__ == '__main__':
  # Avoid the module name being '__main__' instead of main_frame.py
  from tframe.utils.tensor_viewer import main_frame
  # Default file_path
  file_path = None
  init_dir = None
  # file_path = r'E:\rnn_club\98-TOY\records_ms_off\notes\d2_msu(off)3_bs5_lr0.01'
  # file_path += r'\d2_msu(off)3_bs5_lr0.01=1.000.note'
  # init_dir = r'E:/rnn_club/98-TOY/records_ms_off/notes'
  # viewer = main_frame.NoteViewer(note_path=file_path)
  viewer = main_frame.TensorViewer(init_dir=init_dir)
  viewer.show()


