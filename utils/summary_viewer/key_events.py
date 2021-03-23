import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from tframe import console
from . import main_frame as centre


def on_key_press(viewer, event):
  # Sanity check
  assert isinstance(event, tk.Event)
  assert isinstance(viewer, centre.SummaryViewer)

  key_symbol = getattr(event, 'keysym')
  if viewer.in_debug_mode:
    on_key_press_debug(viewer, key_symbol)

  if key_symbol == 'quoteleft':
    console.show_status('Active flags:', symbol='::')
    for k, v in viewer.config_panel.active_config_dict.items():
      console.supplement('{}: {}'.format(k, v), level=2)
  elif key_symbol in ('h', 'k'):
    viewer.header.move_cursor(-1)
  elif key_symbol in ('l', 'j'):
    viewer.header.move_cursor(1)
  elif key_symbol == 'n':
    viewer.criteria_panel.move_between_groups(1)
  elif key_symbol == 'p':
    viewer.criteria_panel.move_between_groups(-1)
  elif key_symbol == 'space':
    viewer.header.show_selected_note_content()
  elif key_symbol == 'Return':
    viewer.header.on_label_detail_click()
  elif key_symbol == 's':
    file_name = tk.filedialog.asksaveasfilename(
      filetypes=[('Note file', '.note')],
      initialfile = 'untitled',
      defaultextension='.note')
    if file_name is not None:
      viewer.header.save_selected_note(file_name)


def load_notes(viewer, _):
  if len(viewer.context.notes) > 0: return
  # Sanity check
  assert isinstance(viewer, centre.SummaryViewer)
  # Select file
  file_path = filedialog.askopenfilename(
    title='Load summary file', filetypes=(('TFrame summary files', '*.sum'),))
  if file_path == '': return
  # Set note file to viewer
  viewer.set_notes(file_path)


def toggle_debug_mode(viewer, _):
  viewer.in_debug_mode = not viewer.in_debug_mode
  if viewer.in_debug_mode:
    bg = 'orange red'
    fg = 'orange'
    # print('>> Debugging mode has been turned on.')
  else:
    bg = 'orange'
    fg = 'orange red'
    # print('>> Debugging mode has been turned off.')
  viewer.set_style(viewer.WidgetNames.TFrame, 'bottom', background=bg)
  viewer.set_style(viewer.WidgetNames.TLabel, 'bottom', background=bg,
                   foreground=fg)


def on_key_press_debug(viewer, key_symbol):
  if key_symbol == 's': _show_widget_size(viewer)
  elif key_symbol == 'd':
    print('[DEBUG] Stop here!')
  elif key_symbol == 'y':
    print('>> y !')
    print(viewer.config_panel.sorted_hyper_list)
  else: print('Unmapped key symbol: {}'.format(key_symbol))


def reload_notes(viewer):
  assert isinstance(viewer, centre.SummaryViewer)
  if viewer.context.summary_file_path is None: return
  if not messagebox.askokcancel('Reload Notes', 'Click OK to reload'): return
  viewer.context.reload()
  # Refresh
  config_panel, criteria_panel = viewer.config_panel, viewer.criteria_panel
  assert isinstance(config_panel, centre.ConfigPanel)
  assert isinstance(criteria_panel, centre.CriteriaPanel)
  config_panel.clear_buffer()
  config_panel.update_combo()
  criteria_panel.clear_buffer()

  viewer.local_refresh()


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





