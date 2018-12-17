from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .base_control import BaseControl

from tframe import console
from tframe.utils.tensor_viewer.main_frame import TensorViewer


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
    # self._notes_buffer = []

    # Package from friends
    # self.package = None

  # region : Properties

  @property
  def config_panel(self):
    return self.master.config_panel

  @property
  def buffer(self):
    return self.master.criteria_panel.notes_buffer
  
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
    num_qualified = len(self.master.config_panel.qualified_notes)
    num_selected = len(self.master.config_panel.matched_notes)
    self.label_notes_info.config(text=self.notes_info.format(
      num_notes, num_qualified, num_selected))

    # Refresh final info label
    self._cursor = 0
    self.config_panel.set_note(self.selected_note)
    self._refresh_detail()

  def show_selected_note_content(self):
    note = self.selected_note
    if note is not None:
      console.show_status('Logs of selected note in header:')
      console.split()
      print(note.content)
      console.split()

  # region : Public Methods

  # region : Private

  def _refresh_detail(self):
    note = self.selected_note
    if note is None:
      self.label_note_detail.configure(text='')
      return

    text = 'Note [{}/{}] '.format(self._cursor + 1, len(self.buffer))
    for i, key in enumerate(self.context.active_criteria_list):
      if i > 0: text += ' | '
      text += '{}: {}'.format(key, self.to_str(note.criteria[key]))
    text += ' '
    self.label_note_detail.configure(text=text)

    # Fancy stuff
    if note.contain_tensors:
      self.label_note_detail.configure(cursor='hand2', foreground='firebrick')
    else: self.label_note_detail.configure(cursor='arrow', foreground='black')

  # endregion : Private

  # region : Events

  def on_label_detail_click(self):
    note = self.selected_note
    if note is not None and note.contain_tensors:
      viewer = TensorViewer(note=note)
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
