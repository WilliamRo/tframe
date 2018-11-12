import tkinter as tk
from tkinter import filedialog

from . import main_frame as centre
from . import variable_viewer


def on_key_press(viewer, event):
  # Sanity check
  assert isinstance(viewer, centre.NoteViewer)
  assert isinstance(event, tk.Event)
  assert isinstance(viewer.variable_viewer, variable_viewer.VariableViewer)

  key_symbol = getattr(event, 'keysym')
  if key_symbol == 'Escape':
    print('|> Note viewer closed.')
    viewer.form.quit()
  elif key_symbol == 'h':
    viewer.loss_figure.on_scroll('scroll', -1, None)
  elif key_symbol == 'l':
    viewer.loss_figure.on_scroll('scroll', 1, None)
  elif key_symbol == 'n':
    viewer.loss_figure.on_scroll('scroll', 1, 'tiny')
  elif key_symbol == 'p':
    viewer.loss_figure.on_scroll('scroll', -1, 'tiny')
  elif key_symbol == 'H':
    viewer.loss_figure.on_scroll('moveto', 0.0)
  elif key_symbol == 'L':
    viewer.loss_figure.on_scroll('moveto', 1.0)
  elif key_symbol == 'j':
    viewer.variable_viewer.next_or_previous(1)
  elif key_symbol == 'k':
    viewer.variable_viewer.next_or_previous(-1)
  elif key_symbol == 'a':
    flag = not viewer.variable_viewer.show_absolute_value
    viewer.variable_viewer.show_absolute_value = flag
    viewer.variable_viewer.refresh()
  elif key_symbol == 'v':
    flag = not viewer.variable_viewer.show_value
    viewer.variable_viewer.show_value = flag
    viewer.variable_viewer.refresh()
  elif key_symbol == 'c':
    flag = not viewer.variable_viewer.use_clim
    viewer.variable_viewer.use_clim = flag
    viewer.variable_viewer.refresh()


def load_note(viewer, _):
  # Sanity check
  assert isinstance(viewer, centre.NoteViewer)
  # Select file
  file_path = filedialog.askopenfilename(
    title='Load note file', filetypes=(('TFrame note files', '*.note'),),
    initialdir=viewer.init_dir,
  )
  if file_path == '': return
  # Set note file to viewer
  viewer.set_note_by_path(file_path)


