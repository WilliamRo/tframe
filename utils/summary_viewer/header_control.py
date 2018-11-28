from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .base_control import BaseControl


class HeaderControl(BaseControl):

  COLOR = 'white'
  notes_info = '[Notes#] Total: {} | Candidates: {} | Selected: {}'
  final_info = 'Final participants: {} '

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.label_notes_info = ttk.Label(self)
    self.label_final_info = ttk.Label(self)


  def load_to_master(self, side=tk.TOP, fill=tk.BOTH, expand=True):
    # Set color
    label_style = self.set_style(
      self.WidgetNames.TLabel, 'header', background=self.COLOR)
    frame_style = self.set_style(
      self.WidgetNames.TFrame, 'header', background=self.COLOR)

    self.label_notes_info.config(style=label_style)
    self.label_final_info.config(style=label_style)
    self.config(style=frame_style)

    # Pack label
    self.label_notes_info.pack(side=tk.LEFT, fill=tk.Y)
    self.label_final_info.pack(side=tk.RIGHT, fill=tk.Y)

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
    num_participants = len(self.master.criteria_panel.final_participants)
    self.label_final_info.config(text=self.final_info.format(num_participants))
