from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import tkinter as tk
  from tkinter.ttk import Frame

  from PIL import Image as Image_
  from PIL import ImageTk

  from tframe.utils.note import Note
  from tframe.utils.note_viewer import key_events
  from tframe.utils.note_viewer.context import Context
  from tframe.utils.note_viewer.loss_figure import LossFigure
  from tframe.utils.note_viewer.variable_viewer import VariableViewer
except Exception as e:
  print(' ! {}'.format(e))
  print(' ! NoteViewer is disabled, install pillow and tkinter to enable it')


class NoteViewer(Frame):
  """Note Viewer for tframe NOTE"""
  SIZE = 400

  def __init__(self, master=None, note_path=None, init_dir=None):
    # If root is provided, load a default one
    if master is None:
      master = tk.Tk()
    # Call parent's initializer
    Frame.__init__(self, master)

    # Layout
    self.loss_figure = None
    self.variable_viewer = None
    self._create_layout()

    # Attributes
    self.form = master
    self.context = Context()
    self.init_dir = init_dir

    # If note_path is provided, try to load it
    if note_path is not None and isinstance(note_path, str):
      self.set_note_by_path(note_path)

    # Initialize viewer
    self._init_viewer()

  # region : Public Methods

  def show(self):
    self.form.after(20, self._move_to_center)
    self.form.mainloop()

  def set_note_by_path(self, note_path):
    # Set context
    self.context.set_note_by_path(note_path)
    # Set loss and variables
    assert isinstance(self.loss_figure, LossFigure)
    self.loss_figure.set_step_and_loss(
      self.context.note.step_array, self.context.note.loss_array)
    # TODO: somehow necessary
    self.loss_figure.refresh()

    assert isinstance(self.variable_viewer, VariableViewer)
    self.variable_viewer.set_variable_dict(self.context.note.variable_dict)

    # Relate loss figure and variable viewer
    self.loss_figure.related_variable_viewer = self.variable_viewer
    self.variable_viewer.related_loss_figure = self.loss_figure

    # Refresh title
    self._refresh()

  # endregion : Public Methods

  # region : Private Methods

  def _init_viewer(self):
    # Bind Key Events
    self.form.bind('<Key>', lambda e: key_events.on_key_press(self, e))
    self.form.bind('<Control-o>', lambda e: key_events.load_note(self, e))

    # Refresh
    self._refresh()

  def _move_to_center(self):
    width = self.master.winfo_width()
    height = self.master.winfo_height()
    width_inc = int((self.master.winfo_screenwidth() - width) / 2)
    height_inc = int((self.master.winfo_screenheight() - height) / 2)
    self.master.geometry(
      '{}x{}+{}+{}'.format(width, height, width_inc, height_inc))

  def _create_layout(self):
    #
    LossFigure.WIDTH = self.SIZE
    LossFigure.HEIGHT = self.SIZE
    self.loss_figure = LossFigure(self)
    self.loss_figure.pack(fill=tk.BOTH, side=tk.LEFT)

    #
    VariableViewer.WIDTH = self.SIZE
    VariableViewer.HEIGHT = self.SIZE
    self.variable_viewer = VariableViewer(self)
    self.variable_viewer.pack(fill=tk.BOTH, side=tk.LEFT, expand=1)

    # Pack self
    self.pack(fill=tk.BOTH, expand=True)

  def _refresh(self):
    # Refresh title
    title = 'Note Viewer'
    if self.context.note_file_name is not None:
      title += ' - {}'.format(self.context.note_file_name)
    self.form.title(title)

  # endregion : Private Methods


if __name__ == '__main__':
  # Avoid the module name being '__main__' instead of main_frame.py
  from tframe.utils.note_viewer import main_frame
  # Default file_path
  file_path = None
  init_dir = None
  # viewer = main_frame.NoteViewer(note_path=file_path)
  viewer = main_frame.NoteViewer(init_dir=init_dir)
  viewer.show()


