import tkinter as tk
from tkinter import filedialog

from . import main_frame as centre
from . import variable_viewer


def on_key_press(viewer, event):
  # Sanity check
  assert isinstance(viewer, centre.TensorViewer)
  assert isinstance(event, tk.Event)
  assert isinstance(viewer.variable_viewer, variable_viewer.VariableViewer)

  key_symbol = getattr(event, 'keysym')
  if key_symbol == 'h':
    viewer.criteria_figure.on_scroll('scroll', -1, None)
  elif key_symbol == 'l':
    viewer.criteria_figure.on_scroll('scroll', 1, None)
  elif key_symbol == 'n':
    viewer.criteria_figure.on_scroll('scroll', 1, 'tiny')
  elif key_symbol == 'p':
    viewer.criteria_figure.on_scroll('scroll', -1, 'tiny')
  elif key_symbol == 'H':
    viewer.criteria_figure.on_scroll('moveto', 0.0)
  elif key_symbol == 'L':
    viewer.criteria_figure.on_scroll('moveto', 1.0)
  elif key_symbol == 'j':
    viewer.variable_viewer.next_or_previous(1, 1)
  elif key_symbol == 'k':
    viewer.variable_viewer.next_or_previous(-1, 1)
  elif key_symbol == 'J':
    viewer.variable_viewer.next_or_previous(1, 0)
  elif key_symbol == 'K':
    viewer.variable_viewer.next_or_previous(-1, 0)
  elif key_symbol == 'a':
    flag = not viewer.variable_viewer.show_absolute_value
    viewer.variable_viewer.show_absolute_value = flag
    viewer.variable_viewer.refresh()
  elif key_symbol == 'v':
    flag = not viewer.variable_viewer.show_value
    viewer.variable_viewer.show_value = flag
    viewer.variable_viewer.refresh()
  elif key_symbol in ['c', 'space']:
    flag = not viewer.variable_viewer.unify_range
    viewer.variable_viewer.unify_range = flag
    viewer.variable_viewer.refresh()
  elif key_symbol == 'g':  # g for loG
    flag = not viewer.variable_viewer.log_scale
    viewer.variable_viewer.log_scale = flag
    viewer.variable_viewer.refresh()
  elif key_symbol == 'e':  # e for export
    flag = not viewer.variable_viewer.for_export
    viewer.variable_viewer.for_export = flag
    viewer.variable_viewer.refresh()
  elif key_symbol == 'E':  # E for 'force real shape'
    flag = not viewer.variable_viewer.force_real_shape
    viewer.variable_viewer.force_real_shape = flag
    viewer.variable_viewer.refresh()
  elif key_symbol == 's':
    # Take a snapshot
    file_name = tk.filedialog.asksaveasfilename(
      filetypes=[('eps file', '.eps')],
      initialfile = 'untitled',
      defaultextension='.csv')
    if file_name is not None:
      viewer.variable_viewer.save_as_eps(file_name)


def load_note(viewer, _):
  # Sanity check
  assert isinstance(viewer, centre.TensorViewer)
  # Select file
  file_path = filedialog.askopenfilename(
    title='Load note file', filetypes=(('TFrame note files', '*.note'),),
    initialdir=viewer.init_dir,
  )
  if file_path == '': return
  # Set note file to viewer
  viewer.set_note(note_path=file_path)


