from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

import numpy as np
from collections import OrderedDict
from tframe import console

from .base_control import BaseControl
from . import main_frame as centre


# region : Decorators

def refresh_friends_at_last(method):
  def wrapper(self, *args, **kwargs):
    assert isinstance(self, CriterionControl)
    method(self, *args, **kwargs)
    self.header.refresh_header()
    assert isinstance(self.criteria_panel, CriteriaPanel)
    self.criteria_panel.refresh()
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

    # Layout
    self.label_frame = ttk.LabelFrame(self)
    self.switch_button = ttk.Button(
      self.label_frame if show else self, cursor='hand2')
    self.statistic_label = ttk.Label(self.label_frame)

    self.find_max_btn = ttk.Button(self.label_frame, cursor='hand2')
    self.find_min_btn = ttk.Button(self.label_frame, cursor='hand2')
    self.avg_btn = ttk.Button(self.label_frame, cursor='hand2')
    self.med_btn = ttk.Button(self.label_frame, cursor='hand2')
    self.detail_button = ttk.Button(self.label_frame, cursor='hand2')

    self._create_layout()

  # region : Properties

  @property
  def value_list(self):
    return [note.criteria[self.name]
            for note in self.criteria_panel.notes_buffer]

  # region : Friends and ancestors

  @property
  def main_frame(self):
    panel = self.criteria_panel.main_frame
    assert isinstance(panel, centre.SummaryViewer)
    return panel

  @property
  def header(self):
    widget = self.main_frame.header
    assert isinstance(widget, centre.HeaderControl)
    return widget

  @property
  def criteria_panel(self):
    panel = self.master.master
    assert isinstance(panel, CriteriaPanel)
    return panel

  @property
  def config_panel(self):
    panel = self.main_frame.config_panel
    assert isinstance(panel, centre.ConfigPanel)
    return panel

  # endregion : Friends and ancestors

  # endregion : Properties

  # region : Public Methods

  def load_to_master(self, side=tk.TOP, fill=tk.BOTH, expand=False):
    if not self._show: side = tk.LEFT
    self.pack(side=side, fill=fill, expand=expand)

  def refresh(self, find_btn_enabled):
    values = self.value_list
    fmt = ' [{}, {}] A: {}, M: {}'
    if len(values) > 0:
      to_str = self.to_str
      val_strs = [to_str(f(values)) for f in (min, max, np.mean, np.median)]
    else: val_strs = ['--'] * 4

    self.statistic_label.config(text=fmt.format(*val_strs))

    # Enable/Disable buttons
    set_btn = lambda btn, enabled: btn.configure(
      state=tk.NORMAL if enabled else tk.DISABLED)

    set_btn(self.find_min_btn, find_btn_enabled)
    set_btn(self.find_max_btn, find_btn_enabled)
    set_btn(self.avg_btn, find_btn_enabled)
    set_btn(self.med_btn, find_btn_enabled)
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
      button_text = '{}'.format(self.name)
      style = self.set_style(
        self.WidgetNames.TButton, 'hidden', self.name, foreground='green',
        width=self.measure(button_text))

    self.switch_button.configure(
      text=button_text, command=self._on_button1_click, style=style)
    self.switch_button.pack(side=tk.LEFT)
    if not self._show: return

    # (2) Statistic label
    self.statistic_label.configure(text='--')
    self.statistic_label.pack(side=tk.LEFT)

    # (3) Detail button
    self.detail_button.configure(text='D', style=self.set_style(
      self.WidgetNames.TButton, 'detail', width=2))
    self.detail_button.configure(command=self._on_detail_btn_click)
    self.detail_button.pack(side=tk.RIGHT)

    # (4) MIN/MAX/AVG/MED buttons
    f_btn_style = self.set_style(self.WidgetNames.TButton, 'fd', width=4)
    def set_search_btn(btn, text, ni, gi):
      btn.configure(text=text, style=f_btn_style,
                    command=lambda: self._on_group_search_btn_click(ni, gi, btn))
      btn.bind(
        '<Button-2>', lambda _: self.criteria_panel.move_between_groups(1, btn))
      btn.bind(
        '<Button-3>', lambda _: self._on_group_search_btn_click(
          ni, -1 - gi * 1, btn))
      btn.pack(side=tk.RIGHT)

    set_search_btn(self.med_btn, 'MED', -1, -1)
    set_search_btn(self.avg_btn, 'AVG', -1, -1)
    set_search_btn(self.find_max_btn, 'MAX', -1, -1)
    set_search_btn(self.find_min_btn, 'MIN', 0, 0)

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

    # Clear buffer
    self.criteria_panel.clear_buffer()

  def _on_detail_btn_click(self):
    if len(self.value_list) == 0:
      console.show_status('No notes with these criteria found under the '
                          'corresponding configures', '::')
      return
    console.show_status('{}:'.format(self.name), '::')
    for v in np.sort(self.value_list):
      console.supplement('{}'.format(v), level=2)

  def _on_group_search_btn_click(self, note_index, group_index, button):
    """Groups, together with their corresponding notes, will be sorted
       without reverse"""
    assert note_index in (-1, 0) and group_index in (-1, 0)
    # Sort groups and each note list
    groups = self.criteria_panel.groups_for_sorting
    num_groups = len(groups)
    assert num_groups > 0
    # Sort members in each group
    for g in groups: g.sort(key=lambda n: n.criteria[self.name])
    # Sort groups
    min_or_max = lambda notes: notes[note_index].criteria[self.name]
    key = {self.find_min_btn: min_or_max,
           self.find_max_btn: min_or_max,
           self.avg_btn: lambda notes: np.mean(
             [n.criteria[self.name] for n in notes]),
           self.med_btn: lambda notes: np.median(
             [n.criteria[self.name] for n in notes])}[button]
    groups.sort(key=key)

    # Set note and refresh corresponding stuff
    note = groups[group_index][note_index]
    self.config_panel.set_note(note)
    self.criteria_panel.notes_buffer = groups[group_index]

    # Make a stamp: (button, groups, multiplier, index, note_index)
    index = group_index if group_index >= 0 else group_index + len(groups)
    self.criteria_panel.button_stamp = (
      button, groups, 1 if group_index == 0 else -1, index, note_index)

    # Finally refresh header
    self.header.refresh_header()

  # endregion : Events


class CriteriaPanel(BaseControl):

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.stat_panel = ttk.LabelFrame(self, labelanchor=tk.S)
    self.hidden_panel = ttk.LabelFrame(self, text='Hidden Criteria')

    # Attributes
    self.explicit_dict = OrderedDict()
    self.hidden_dict = OrderedDict()

    # Buffers for faster sorting
    self._candidates_set = None
    self._notes_buffer = None
    self.button_stamp = None

  # region : Properties

  @property
  def groups_for_sorting(self):
    groups = [self.criteria_filter(note_list)
              for note_list in self.config_panel.selected_group_values]
    return [g for g in groups if len(g) > 0]

  @property
  def notes_buffer(self):
    if self._notes_buffer is None:
      self._notes_buffer = self.criteria_filter(self.config_panel.matched_notes)
    return self._notes_buffer

  @notes_buffer.setter
  def notes_buffer(self, val):
    assert isinstance(val, list) and len(val) > 0
    self._notes_buffer = val

  @property
  def notes_with_active_criteria(self):
    if self._candidates_set is None:
      self._candidates_set = set([
        note for note in self.context.notes
        if set(note.criteria.keys()).issuperset(
          self.context.active_criteria_set)
      ])
    return self._candidates_set

  @property
  def minimum_height(self):
    h_hidden = 48
    h_each_control = 48
    return h_hidden + len(self.explicit_dict) * h_each_control

  # region : Friends and ancestors

  @property
  def main_frame(self):
    frame = self.master.master
    assert isinstance(frame, centre.SummaryViewer)
    return frame

  @property
  def header(self):
    control = self.main_frame.header
    assert isinstance(control, centre.HeaderControl)
    return control

  @property
  def config_panel(self):
    panel = self.main_frame.config_panel
    assert isinstance(panel, centre.ConfigPanel)
    return panel

  # endregion : Friends and ancestors

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
    for k in self.context.active_criteria_list:
      self.explicit_dict[k].load_to_master()
    for k in self.context.inactive_criteria_list:
      self.hidden_dict[k].load_to_master()

  def refresh(self):
    min_max_btn_enabled = len(self.groups_for_sorting) > 0
    self._notes_buffer = None
    self.button_stamp = None
    # Refresh each explicit criteria control
    for active_criterion in self.context.active_criteria_set:
      # Get widget
      criterion_control = self.explicit_dict[active_criterion]
      assert isinstance(criterion_control, CriterionControl)
      # Refresh widget
      criterion_control.refresh(min_max_btn_enabled)

  def clear_buffer(self):
    self._candidates_set = None
    self._notes_buffer = None
    self.button_stamp = None

  def criteria_filter(self, note_set):
    return list(set(note_set).intersection(self.notes_with_active_criteria))

  # endregion : Public Methods

  # region : Private Methods


  # endregion : Private Methods

  # region : Fancy stuff

  def move_between_groups(self, offset, button=None):
    """stamp format: (button, groups, group_index, index)"""
    # Sanity checks and unwrap
    if self.button_stamp is None: return
    last_button, groups, multiplier, index, note_index = self.button_stamp
    if button is not None and  button is not last_button: return
    offset *= multiplier
    assert offset in (-1, 1) and note_index in (-1, 0)

    # Set note and refresh corresponding stuff
    index += offset
    if index < 0: index += len(groups)
    if index >= len(groups): index = 0
    note = groups[index][note_index]
    self.config_panel.set_note(note)
    self.notes_buffer = groups[index]

    # Set stamp
    self.button_stamp = (button, groups, multiplier, index, note_index)

    # Finally refresh header
    self.header.refresh_header()

  # endregion : Fancy stuff


