from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import tkinter as tk
  from tkinter.ttk import Frame

  from PIL import Image as Image_
  from PIL import ImageTk

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
  SIZE = 500

  def __init__(self, master=None, note_path=None, init_dir=None):
    # Call parent's initializer
    Viewer.__init__(self, master)
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
    self._define_layout()
    self._global_refresh()

    # If note_path is provided, try to load it
    if note_path is not None and isinstance(note_path, str):
      self.set_note_by_path(note_path)

  # region : Public Methods

  def set_note_by_path(self, note_path):
    # Set context
    self.context.set_note_by_path(note_path)
    # Set loss and variables
    assert isinstance(self.criteria_figure, CriteriaFigure)
    self.criteria_figure.set_step_and_loss(
      self.context.note.step_array, self.context.note.loss_array)
    # TODO: somehow necessary
    self.criteria_figure.refresh()

    assert isinstance(self.variable_viewer, VariableViewer)
    self.variable_viewer.set_variable_dict(self.context.note.variable_dict)

    # Relate loss figure and variable viewer
    self.criteria_figure.related_variable_viewer = self.variable_viewer
    self.variable_viewer.related_loss_figure = self.criteria_figure

    # Global refresh
    self._global_refresh()

  # endregion : Public Methods

  # region : Private Methods

  def _bind_key_events(self):
    # Bind Key Events
    self.form.bind('<Key>', lambda e: key_events.on_key_press(self, e))
    self.form.bind('<Control-o>', lambda e: key_events.load_note(self, e))

  def _define_layout(self):
    #
    CriteriaFigure.WIDTH = self.SIZE
    CriteriaFigure.HEIGHT = self.SIZE
    self.criteria_figure = CriteriaFigure(self)
    self.criteria_figure.pack(fill=tk.BOTH, side=tk.LEFT)

    #
    VariableViewer.WIDTH = self.SIZE
    VariableViewer.HEIGHT = self.SIZE
    self.variable_viewer = VariableViewer(self)
    self.variable_viewer.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)

    # Pack self
    self.pack(fill=tk.BOTH, expand=True)

  def _global_refresh(self):
    # Refresh title
    title = 'Note Viewer'
    if self.context.note_file_name is not None:
      title += ' - {}'.format(self.context.note_file_name)
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


