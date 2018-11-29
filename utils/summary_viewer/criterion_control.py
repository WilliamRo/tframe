from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

import numpy as np
from tframe import console

from .context import Context
from .base_control import BaseControl


# region : Decorators

def refresh_friends_at_last(method):
  def wrapper(self, *args, **kwargs):
    assert isinstance(self, CriterionControl)
    method(self, *args, **kwargs)
    self.header.refresh()
  return wrapper

# endregion : Decorators


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

    # Ancestors and friends
    self.criteria_panel = self.master.master
    assert isinstance(self.criteria_panel, CriteriaPanel)
    self.main_frame = self.criteria_panel.main_frame
    self.config_panel = self.main_frame.config_panel
    self.header = self.main_frame.header

    # Layout
    self.label_frame = ttk.LabelFrame(self)
    self.switch_button = ttk.Button(
      self.label_frame if show else self, cursor='hand2')
    self.statistic_label = ttk.Label(self.label_frame)

    self.find_max_btn = ttk.Button(self.label_frame, cursor='hand2')
    self.find_min_btn = ttk.Button(self.label_frame, cursor='hand2')
    self.detail_button = ttk.Button(self.label_frame, cursor='hand2')

    self._create_layout()

  # region : Properties

  @property
  def value_list(self):
    return [note.criteria[self.name]
            for note in self.criteria_panel.final_participants]

  # endregion : Properties

  # region : Public Methods

  def load_to_master(self, side=tk.TOP, fill=tk.BOTH, expand=False):
    if not self._show: side = tk.LEFT
    self.pack(side=side, fill=fill, expand=expand)

  def refresh(self, btn_enabled):
    values = self.value_list
    fmt = '  Avg: {},  Range: [{}, {}]'
    if len(values) > 0:
      to_str = self.to_str
      min_v, max_v = min(values), max(values)
      p0, p1, p2 = to_str(np.mean(values)), to_str(min_v), to_str(max_v)
    else: p0, p1, p2 = ('--',) * 3

    self.statistic_label.config(text=fmt.format(p0, p1, p2))

    # Enable/Disable buttons
    set_btn = lambda btn, enabled: btn.configure(
      state=tk.NORMAL if enabled else tk.DISABLED)

    set_btn(self.find_min_btn, btn_enabled)
    set_btn(self.find_max_btn, btn_enabled)
    set_btn(self.detail_button, len(values) > 0)

  # endregion : Public Methods

  # region : Private Methods

  def _create_layout(self):
    # (1) Button
    if self._show:
      style = self.set_style(
        self.WidgetNames.TButton, 'explicit', foreground='red', width=5)
      button_text = 'Hide'
    else:
      button_text = 'Show `{}`'.format(self.name)
      style = self.set_style(
        self.WidgetNames.TButton, 'hidden', foreground='green')

    self.switch_button.configure(
      text=button_text, command=self._on_button1_click, style=style)
    self.switch_button.pack(side=tk.LEFT)
    if not self._show: return

    # (2) Statistic label
    self.statistic_label.configure(text='--')
    self.statistic_label.pack(side=tk.LEFT)

    # (3) Detail button
    self.detail_button.configure(text='Detail', style=self.set_style(
      self.WidgetNames.TButton, 'detail', width=6))
    self.detail_button.configure(command=self._on_detail_btn_click)
    self.detail_button.pack(side=tk.RIGHT)

    # (4) Find max & min button
    f_btn_style = self.set_style(self.WidgetNames.TButton, 'fd', width=4)
    self.find_min_btn.configure(
      text='FMI',
      style=f_btn_style, command=lambda: self._on_find_btn_click(False))
    self.find_max_btn.configure(
      text='FMA',
      style=f_btn_style, command=lambda: self._on_find_btn_click(True))
    self.find_max_btn.pack(side=tk.RIGHT)
    self.find_min_btn.pack(side=tk.RIGHT)

    # (9) Label frame
    self.label_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.label_frame.configure(text=self.name)

  # endregion : Private Methods

  # region : Events

  @refresh_friends_at_last
  def _on_button1_click(self):
    # Hide this control
    self.pack_forget()
    # Show the corresponding control
    if self._show:
      op_control = self.criteria_panel.hidden_dict[self.name]
      src_set = self.context.active_criteria_set
      tgt_set = self.context.inactive_criteria_set
    else:
      op_control = self.criteria_panel.explicit_dict[self.name]
      src_set = self.context.inactive_criteria_set
      tgt_set = self.context.active_criteria_set
    assert isinstance(op_control, CriterionControl)
    op_control.load_to_master()

    # Modify sets in context
    src_set.remove(self.name)
    tgt_set.add(self.name)

  def _on_detail_btn_click(self):
    if len(self.value_list) == 0:
      console.show_status('No notes with these criteria found under the '
                          'corresponding configures', '::')
      return
    console.show_status('{}:'.format(self.name), '::')
    for v in np.sort(self.value_list):
      console.supplement('{}'.format(v), level=2)

  def _on_find_btn_click(self, find_max):
    # Find the corresponding note
    notes = self.criteria_panel._filter(self.config_panel.qualified_notes)
    assert isinstance(notes, list) and len(notes) > 0
    notes.sort(key=lambda n: n.criteria[self.name], reverse=find_max)
    note = notes[0]

    # Print note's
    val_str = self.to_str(note.criteria[self.name])
    console.show_status( 'Logs of note with {} `{}`({}):'.format(
      'max' if find_max else 'min', self.name, val_str), '::')
    splitter = '-' * 79
    print('- ' * 40)
    print(note.content)
    print(splitter)

    # Set note to config_panel
    self.config_panel.set_note(note)

  # endregion : Events


class CriteriaPanel(BaseControl):

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.stat_panel = ttk.LabelFrame(self, labelanchor=tk.S)
    self.hidden_panel = ttk.LabelFrame(self, text='Hidden Criteria')

    # Attributes
    self.explicit_dict = {}
    self.hidden_dict = {}

    # Ancestor and friends
    self.main_frame = self.master.master
    self.header = self.main_frame.header
    self.config_panel = self.main_frame.config_panel

  # region : Properties

  @property
  def final_participants(self):
    return self._filter(self.config_panel.selected_notes)

  # endregion : Properties

  # region : Public Methods

  def load_to_master(self, side=tk.RIGHT, fill=tk.BOTH, expand=True):
    # Pack panels
    pack_params = {'fill': tk.BOTH, 'side': tk.TOP, 'expand': True}
    for label_frame in (self.stat_panel, self.hidden_panel):
      label_frame.configure(width=400, height=48)
      label_frame.pack(**pack_params)
    self.hidden_panel.pack(expand=False)

    # Pack self
    self.pack(fill=fill, side=side, expand=expand)

  def initialize_criteria_controls(self):
    # TODO: Reloading is not allowed
    # Create controls
    for k in self.context.active_criteria_set.union(
      self.context.inactive_criteria_set):
      # Create an active one
      self.explicit_dict[k] = CriterionControl(self.stat_panel, k, True)
      # Create an inactive one
      self.hidden_dict[k] = CriterionControl(self.hidden_panel, k, False)

    # Pack criteria controls
    for k in self.context.active_criteria_set:
      self.explicit_dict[k].load_to_master()
    for k in self.context.inactive_criteria_set:
      self.hidden_dict[k].load_to_master()

  def refresh(self):
    btn_enabled = len(self.config_panel.qualified_notes) > 0
    # Refresh each explicit criteria control
    for e_c in self.explicit_dict.values():
      assert isinstance(e_c, CriterionControl)
      e_c.refresh(btn_enabled)

  # endregion : Public Methods

  # region : Private Methods

  def _filter(self, note_set):
    return [
      note for note in note_set
      if set(note.criteria.keys()).issuperset(set(self.explicit_dict.keys()))]

  # endregion : Private Methods


