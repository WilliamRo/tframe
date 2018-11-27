from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .context import Context
from .base_control import BaseControl


class CriterionControl(BaseControl):
  # Basic configurations
  WIDTH = 400
  HEIGHT = 100

  def __init__(self, master, name, show):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Attributes
    self.name = name
    self._show = show

    # Ancestors
    self.criteria_panel = self.master.master
    assert isinstance(self.criteria_panel, CriteriaPanel)

    # Layout
    self.switch_button = ttk.Button(self, cursor='hand2')
    self.statistic_label = ttk.Label(self)
    self.detail_button = ttk.Button(self, cursor='hand2')
    self._create_layout()

  # region : Properties

  # endregion : Properties

  # region : Public Methods

  def load_to_master(self, side=tk.TOP, fill=tk.BOTH, expand=True):
    self.pack(side=side, fill=fill, expand=expand)

  # endregion : Public Methods

  # region : Private Methods

  def _create_layout(self):
    # (1) Button
    button_text = 'Hide' if self._show else self.name
    self.switch_button.configure(
      text=button_text, command=self._on_button_click)
    self.switch_button.pack(side=tk.LEFT)
    if not self._show: return

    # (2) Statistic label
    self.statistic_label.configure(text='--')
    self.statistic_label.pack(side=tk.LEFT)

    # (3) Detail button
    self.detail_button.configure(text='Detail')
    self.detail_button.pack(side=tk.RIGHT)

  # endregion : Private Methods

  # region : Events

  def _on_button_click(self):
    pass

  # endregion : Events


class CriteriaPanel(BaseControl):

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.stat_panel = ttk.LabelFrame(self, text='Statistics')
    self.hidden_panel = ttk.LabelFrame(self, text='Hidden Stat')

    # Attributes
    self.explicit_dict = {}
    self.hidden_dict = {}

    # Ancestor
    self.main_frame = self.master.master

  # region : Public Methods

  def load_to_master(self, side=tk.RIGHT, fill=tk.BOTH, expand=True):
    pass

  def refresh(self):
    pass

  # endregion : Public Methods

  # region : Private Methods

  # endregion : Private Methods


