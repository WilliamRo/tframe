from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .base_control import BaseControl


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
    self._notes_buffer = []

    # Package from friends
    self.package = None

  # region : Properties
  
  @property
  def final_participants(self):
    return self.master.criteria_panel.final_participants
  
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

    # Pack self
    self.pack(fill=fill, side=side, expand=expand)

  def refresh(self):
    # Refresh basic info label
    num_notes = len(self.context.notes)
    num_qualified = len(self.master.config_panel.qualified_notes)
    num_selected = len(self.master.config_panel.selected_notes)
    self.label_notes_info.config(text=self.notes_info.format(
      num_notes, num_qualified, num_selected))

    # Refresh final info label
    self._cursor = 0
    self._set_note_buffer()
    # self._notes_buffer = self.final_participants
    self._refresh_detail()

  # region : Public Methods

  # region : Private

  def _set_note_buffer(self):
    self._notes_buffer = self.final_participants
    if self.package is None: return
    key, reverse = self.package
    self._notes_buffer.sort(key=lambda n: n.criteria[key], reverse=reverse)
    self.package = None

  def _refresh_detail(self):
    if len(self._notes_buffer) == 0:
      self.label_note_detail.configure(text='')
      return

    text = 'Note [{}/{}] '.format(self._cursor + 1, len(self._notes_buffer))
    for i, key in enumerate(self.context.active_criteria_set):
      if i > 0: text += ' | '
      text += '{}: {}'.format(key, self.to_str(
        self._notes_buffer[self._cursor].criteria[key]))
    text += ' '
    self.label_note_detail.configure(text=text)

  # endregion : Private

  # region : Events

  def move_cursor(self, offset):
    assert offset in (-1, 1)
    total = len(self._notes_buffer)
    if total == 0: return
    cursor = self._cursor
    cursor += offset
    if cursor < 0: cursor += total
    elif cursor >= total: cursor -= total
    if cursor != self._cursor:
      self._cursor = cursor
      self._refresh_detail()

  # endregion : Events
