from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .base_control import BaseControl

from tframe import console
from tframe.utils.tensor_viewer.main_frame import TensorViewer
from . import main_frame as centre


class HeaderControl(BaseControl):

  COLOR = 'white'
  notes_info = '[Notes#] Total: {} | Candidates: {} | Selected: {}'

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.label_notes_info = ttk.Label(self)
    self.label_note_detail = ttk.Label(self)

    # Attributes
    self._cursor = 0

  # region : Properties

  @property
  def main_frame(self):
    frame = self.master
    assert isinstance(frame, centre.SummaryViewer)
    return frame

  @property
  def config_panel(self):
    panel = self.main_frame.config_panel
    assert isinstance(panel, centre.ConfigPanel)
    return panel

  @property
  def buffer(self):
    return self.main_frame.criteria_panel.notes_buffer

  @property
  def selected_note(self):
    if self.buffer: return self.buffer[self._cursor]
    else: return None

  # endregion : Properties

  # region : Public Methods

  def load_to_master(self, side=tk.TOP, fill=tk.BOTH, expand=True):
    # Set color
    label_style = self.set_style(
      self.WidgetNames.TLabel, 'header', background=self.COLOR)
    frame_style = self.set_style(
      self.WidgetNames.TFrame, 'header', background=self.COLOR)

    self.label_notes_info.config(style=label_style)
    self.label_note_detail.config(style=label_style)
    self.config(style=frame_style)

    # Pack label
    self.label_notes_info.pack(side=tk.LEFT, fill=tk.Y)
    self.label_note_detail.pack(side=tk.RIGHT, fill=tk.Y)
    self.label_note_detail.bind(
      '<Button-1>', lambda _: self.on_label_detail_click())

    # Pack self
    self.pack(fill=fill, side=side, expand=expand)

  def refresh_header(self):
    # Refresh basic info label
    num_notes = len(self.context.notes)
    num_qualified = len(self.config_panel.qualified_notes)
    num_selected = len(self.config_panel.matched_notes)
    self.label_notes_info.config(text=self.notes_info.format(
      num_notes, num_qualified, num_selected))

    # Refresh final info label
    self._cursor = 0
    stamp = self.main_frame.criteria_panel.button_stamp
    if stamp is not None:
      _, groups, _, index, note_index = stamp
      assert note_index in (0, -1)
      if note_index == -1:
        self._cursor = len(self.buffer) - 1
    self.config_panel.set_note(self.selected_note)
    self._refresh_detail()

  def show_selected_note_content(self):
    note = self.selected_note
    if note is not None:
      console.show_status('Logs of selected note in header:')
      console.split()
      print(note.content)
      console.split()

  def save_selected_note(self, file_name):
    from tframe.utils.note import Note
    assert isinstance(file_name, str)
    note = self.selected_note
    assert isinstance(note, Note)
    note.save(file_name)
    console.show_status('Note saved to `{}`'.format(file_name))

  # region : Public Methods

  # region : Private

  def _refresh_detail(self):
    note = self.selected_note
    if note is None:
      self.label_note_detail.configure(text='')
      return

    # Insert group info
    text = ''
    stamp = self.main_frame.criteria_panel.button_stamp
    if stamp is not None:
      _, groups, _, index, _ = stamp
      text += 'Groups [{}/{}]'.format(index + 1, len(groups))
    else:
      text += '{} Groups -'.format(
        len(self.main_frame.criteria_panel.groups_for_sorting))

    text += ' Note [{}/{}] '.format(self._cursor + 1, len(self.buffer))
    for i, key in enumerate(self.context.active_criteria_list):
      if i > 0: text += ' | '
      text += '{}: {}'.format(key, self.to_str(note.criteria[key]))
    text += ' '
    self.label_note_detail.configure(text=text)

    # Fancy stuff
    if note.has_history:
      self.label_note_detail.configure(cursor='hand2', foreground='firebrick')
    else: self.label_note_detail.configure(cursor='arrow', foreground='black')

  # endregion : Private

  # region : Events

  def on_label_detail_click(self):
    note = self.selected_note
    if note is not None and note.has_history:
      viewer = TensorViewer(note=note, plugins=self.main_frame.plugins)
      viewer.show()

  def move_cursor(self, offset):
    assert offset in (-1, 1)
    total = len(self.buffer)
    if total == 0: return
    cursor = self._cursor
    cursor += offset
    if cursor < 0: cursor += total
    elif cursor >= total: cursor -= total
    if cursor != self._cursor:
      self._cursor = cursor
      self.config_panel.set_note(self.selected_note)
      self._refresh_detail()

  # endregion : Events
