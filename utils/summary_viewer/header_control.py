from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .base_control import BaseControl


class HeaderControl(BaseControl):

  notes_info = '[Notes#] Total: {} | Candidates: {} | Selected: {}'

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.label_notes_info = ttk.Label(self)


  def load_to_master(self, side=tk.TOP, fill=tk.BOTH, expand=True):
    # Pack label
    self.label_notes_info.pack(side=tk.LEFT, fill=tk.Y)

    # Pack self
    self.pack(fill=fill, side=side, expand=expand)


  def refresh(self):
    # Refresh basic info label
    num_notes = len(self.context.notes)
    num_qualified = len(self.master.config_panel.qualified_notes)
    num_selected = len(self.master.config_panel.selected_notes)
    self.label_notes_info.config(text=self.notes_info.format(
      num_notes, num_qualified, num_selected))
