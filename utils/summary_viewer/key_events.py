import tkinter as tk
from tkinter import filedialog

from tframe import console
from . import main_frame as centre


def on_key_press(viewer, event):
  # Sanity check
  assert isinstance(event, tk.Event)

  key_symbol = getattr(event, 'keysym')
  if key_symbol == 't':
    print('[TEST] Hello')
  elif key_symbol == 'quoteleft':
    console.show_status('Active flags:', symbol='::')
    for k, v in viewer.config_panel.active_config_dict.items():
      console.supplement('{}: {}'.format(k, v), level=2)


def load_notes(viewer, _):
  if len(viewer.context.notes) > 0: return
  # Sanity check
  assert isinstance(viewer, centre.SummaryViewer)
  # Select file
  file_path = filedialog.askopenfilename(
    title='Load summary file', filetypes=(('TFrame summary files', '*.sum'),))
  if file_path == '': return
  # Set note file to viewer
  viewer.set_notes_by_path(file_path)


