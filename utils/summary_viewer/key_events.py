import tkinter as tk
from tkinter import filedialog

from tframe import console
from . import main_frame as centre


def on_key_press(viewer, event):
  # Sanity check
  assert isinstance(event, tk.Event)

  key_symbol = getattr(event, 'keysym')
  if viewer.in_debug_mode:
    on_key_press_debug(viewer, key_symbol)
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


def toggle_debug_mode(viewer, _):
  viewer.in_debug_mode = not viewer.in_debug_mode
  if viewer.in_debug_mode:
    bg = 'orange red'
    print('>> Debugging mode has been turned on.')
  else:
    bg = 'orange'
    print('>> Debugging mode has been turned off.')
  viewer.set_style(viewer.WidgetNames.TFrame, 'bottom', background=bg)


def on_key_press_debug(viewer, key_symbol):
  if key_symbol == 's': _show_widget_size(viewer)
  elif key_symbol == 'd':
    print('[DEBUG] Stop here!')
  else: print('Unmapped key symbol: {}'.format(key_symbol))


def _show_widget_size(widget, level=0):
  def show_size(n, o):
    print('{}: H({})xW({})'.format(n, o.winfo_height(), o.winfo_width()))
  # Show root info if necessary
  if level == 0: show_size('ROOT', widget.master)
  # Show the size of widget
  spaces = '..' * level
  name = '{}{}'.format(spaces, widget.__class__.__name__)
  show_size(name, widget)
  # If widget has children, show all their sizes
  if level >= 4: return  # Details are not interested
  for child in widget.children.values():
    _show_widget_size(child, level + 1)





