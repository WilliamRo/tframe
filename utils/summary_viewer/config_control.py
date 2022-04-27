from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tkinter as tk
import tkinter.ttk as ttk

from collections import OrderedDict
from .base_control import BaseControl
from . import main_frame as centre


# region : Decorators

def refresh_friends_at_last(force_not_empty=False):
  def outer_wrapper(method):
    def inner_wrapper(self, *args, **kwargs):
      assert isinstance(self, (ConfigControl, ConfigPanel))
      method(self, *args, **kwargs)
      self.criteria_panel.refresh()
      self.header.refresh_header()
      if force_not_empty:
        self.main_frame.force_buffer_not_empty()
    return inner_wrapper
  return outer_wrapper

# endregion : Decorators


class ConfigControl(BaseControl):

  # Basic configurations
  WIDTH = 400
  HEIGHT = 100
  MAX_STR_LENGTH = 20

  def __init__(self, master, flag_name, flag_values, is_active):
    # Sanity check
    assert isinstance(flag_name, str) and isinstance(flag_values, (tuple, list))
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Attributes
    self.name = flag_name
    assert len(flag_values) > 0
    self.all_values = flag_values
    self.values = flag_values
    assert isinstance(is_active, bool)
    self._active = is_active
    self.fixed = False

    # Layout
    self.switch_button = ttk.Button(self, cursor='hand2')
    self.label_name = ttk.Label(self)
    self.values_control = None
    self._create_layout()

  # region : Properties

  @property
  def is_common(self):
    return len(self.all_values) == 1

  @property
  def current_value(self):
    if self.is_common:
      return self.all_values[0]
    else:
      assert isinstance(self.values_control, ttk.Combobox)
      return self.values[self.values_control.current()]

  # region : Friends and ancestors

  @property
  def config_panel(self):
    panel = self.master.master
    assert isinstance(panel, ConfigPanel)
    return panel

  @property
  def main_frame(self):
    return self.config_panel.main_frame

  @property
  def criteria_panel(self):
    return self.config_panel.criteria_panel

  @property
  def header(self):
    return self.main_frame.header

  # endregion : Friends and ancestors

  # endregion : Properties

  # region : Public Methods

  def load_to_master(self, side=tk.TOP, fill=tk.X, expand=False):
    self.pack(side=side, fill=fill, expand=expand)

  def refresh(self):
    pass

  def set_value(self, val):
    if self.current_value == val: return False
    index = self.values.index(val)
    assert index >= 0 and isinstance(self.values_control, ttk.Combobox)
    self.values_control.current(index)
    return True

  def reset_combo(self, values):
    assert isinstance(values, (list, tuple))
    # Save current value
    old_val = self.current_value
    # Try to set value back to combobox
    self.values_control.config(values=values)
    if old_val in values:
      self.values_control.current(values.index(old_val))
    else: self.values_control.current(0)
    self.values = values

  def indicate_possession(self, possess):
    assert possess in (True, False)
    if possess:
      style = self.set_style(self.WidgetNames.TLabel, 'possess',
                             foreground='black')
    else:
      style = self.set_style(self.WidgetNames.TLabel, 'not_possess',
                             foreground='grey')
    self.label_name.configure(style=style)

  # endregion : Public Methods

  # region : Events

  @refresh_friends_at_last()
  def _on_combobox_selected(self, _):
    if not self._active: return

  @refresh_friends_at_last(force_not_empty=True)
  def _on_button_click(self):
    # Hide this control
    self.pack_forget()
    # Show the corresponding control
    panel = self.config_panel

    if self._active:
      op_control = panel.inactive_dict[self.name]
      src_set = self.context.active_flag_set
      tgt_set = self.context.inactive_flag_set
    else:
      op_control = panel.active_dict[self.name]
      src_set = self.context.inactive_flag_set
      tgt_set = self.context.active_flag_set

    assert isinstance(op_control, ConfigControl)
    op_control.load_to_master()

    # Modify sets in context
    src_set.remove(self.name)
    tgt_set.add(self.name)

    # Clear buffer
    self.config_panel.clear_buffer()
    # Update combo
    self.config_panel.update_combo()

  @refresh_friends_at_last()
  def _move_combo_cursor(self, offset):
    assert offset in (-1, 1) and isinstance(self.values_control, ttk.Combobox)
    if self.fixed: return
    total = len(self.values)
    index = self.values_control.current() + offset
    if index < 0: index += total
    elif index >= total: index -= total
    self.values_control.current(index)

  def _lock_or_unlock(self):
    assert isinstance(self.values_control, ttk.Combobox)
    if self.fixed:
      self.fixed = False
      self.values_control.configure(state='readonly')
      self.label_name.configure(cursor='double_arrow')
    else:
      self.fixed = True
      self.values_control.configure(state=tk.DISABLED)
      self.label_name.configure(cursor='circle')

  # endregion : Events

  # region : Private Methods

  def _create_layout(self):
    # (1) Button
    if self._active:
      style = self.set_style(
        self.WidgetNames.TButton, 'active', foreground='red')
      text = 'Deactivate'
    else:
      style = self.set_style(
        self.WidgetNames.TButton, 'inactive', foreground='green')
      text = 'Activate'
    self.switch_button.configure(
      style=style, text=text, command=self._on_button_click)
    self.switch_button.pack(side=tk.LEFT)

    # (2) Label
    self.label_name.configure(text=' {}:'.format(self.name))
    self.label_name.pack(side=tk.LEFT)
    if self._active and not self.is_common:
      self.label_name.config(cursor='double_arrow')
      self.label_name.bind('<Button-1>', lambda _: self._move_combo_cursor(1))
      self.label_name.bind('<Button-2>', lambda _: self._lock_or_unlock())
      self.label_name.bind('<Button-3>', lambda _: self._move_combo_cursor(-1))

    # (3) Value
    if self.is_common:
      # TODO: temporary patch to handle long label text issue
      value = self.all_values[0]
      if not isinstance(value, str):
        value = str(value)
        m = re.match(r"<class '([\w]+.)+([\w]+)'>", value)
        if m is not None: value = m.group(1)

      # Truncate long text if necessary
      text = value
      if len(text) > self.MAX_STR_LENGTH:
        text = text[:self.MAX_STR_LENGTH] + '...'
      self.values_control = ttk.Label(self, text=text)
    else:
      self.values_control = ttk.Combobox(
        self, state='readonly', justify=tk.RIGHT)
      self.values_control.config(values=self.all_values)
      self.values_control.current(0)
      if self._active:
        self.values_control.bind(
          '<<ComboboxSelected>>', self._on_combobox_selected)
    self.values_control.pack(side=tk.RIGHT)

  # endregion : Private Methods


class ConfigPanel(BaseControl):

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.hyper_frame = ttk.LabelFrame(self, text='Hyper-Parameters')
    self.common_frame = ttk.LabelFrame(self, text='Common Configurations')
    self.inactive_frame = ttk.LabelFrame(self, text='Inactive Configurations')

    # Attributes
    self.active_dict = OrderedDict()
    self.inactive_dict = OrderedDict()

    # Buffers for faster sorting
    self._candidates = None
    self._groups = None
    self._sorted_hyper = None

  # region : Properties

  @property
  def active_config_dict(self):
    return {k: self.active_dict[k].current_value
            for k in self.context.active_flag_set}

  @property
  def sorted_hyper_list(self):
    if self._sorted_hyper is not None: return self._sorted_hyper
    hypers = [name for name, widget in self.active_dict.items()
              if name in self.context.active_flag_set and not widget.is_common]
    hypers.sort()
    self._sorted_hyper = hypers
    return self._sorted_hyper
  
  @property
  def qualified_notes(self):
    if self._candidates is not None: return self._candidates
    flag_of_interest = set(self.active_config_dict.keys())
    self._candidates = [
      note for note in self.context.notes
      if set(note.configs.keys()).issuperset(flag_of_interest)]
    return self._candidates

  @property
  def groups(self):
    if self._groups is not None: return self._groups
    self._groups = OrderedDict()
    for note in self.qualified_notes:
      key = self._get_config_tuple(note)
      if key not in self._groups.keys(): self._groups[key] = []
      self._groups[key].append(note)
    return self._groups

  @property
  def selected_group_values(self):
    """Groups(buffered) filtered according to fixed configs"""
    fixed_config_set = set(
      [(k, self.active_dict[k].current_value)
       for k in self.sorted_hyper_list if self.active_dict[k].fixed])

    if len(fixed_config_set) == 0:
      return tuple(self.groups.values())
    else:
      return tuple([v for k, v in self.groups.items()
                    if set(k).issuperset(fixed_config_set)])

  @property
  def matched_notes(self):
    return self.groups.get(self._get_config_tuple(), [])
    # return self._filter(self.qualified_notes, self.active_config_dict)

  @property
  def minimum_height(self):
    h_empty_panel = 21
    h_each_control = 27
    coef = 3 if len(self.active_dict) == 0 else 2
    return 3 * h_empty_panel + (coef + len(self.active_dict)) * h_each_control

  # region : Friends and ancestors
  
  @property
  def main_frame(self):
    frame = self.master.master
    assert isinstance(frame, centre.SummaryViewer)
    return frame

  @property
  def criteria_panel(self):
    panel = self.main_frame.criteria_panel
    assert isinstance(panel, centre.CriteriaPanel)
    return panel

  @property
  def header(self):
    control = self.main_frame.header
    assert isinstance(control, centre.HeaderControl)
    return control

  # endregion : Friends and ancestors

  # endregion : Properties

  # region : Public Methods

  def initialize_config_controls(self):
    # self.inactive_dict, self.active_dict = {}, {}
    for k, v in self.context.flag_value_dict.items():
      # Create an active one
      master = self.hyper_frame if len(v) > 1 else self.common_frame
      active_control = ConfigControl(master, k, v, True)
      self.active_dict[k] = active_control
      # Create an inactive one
      inactive_control = ConfigControl(self.inactive_frame, k, v, False)
      self.inactive_dict[k] = inactive_control

    # Clear 3 panels TODO: reloading is not allowed
    # self.hyper_parameters.

    # Pack config controls
    for k in self.context.active_flag_list:
      control = self.active_dict[k]
      assert isinstance(control, ConfigControl)
      control.load_to_master()
    for k in self.context.inactive_flag_list:
      control = self.inactive_dict[k]
      assert isinstance(control, ConfigControl)
      control.load_to_master()


  def load_to_master(self, side=tk.LEFT, fill=tk.Y, expand=True):
    # Pack label-frames
    pack_params = {'fill': tk.BOTH, 'side': tk.TOP, 'expand': False}
    for label_frame in (
        self.hyper_frame,
        self.common_frame,
        self.inactive_frame,
    ):
      label_frame.configure(width=400, height=48)
      label_frame.pack(**pack_params)
    self.inactive_frame.pack(expand=True)
    self.inactive_frame.configure()

    # Pack self
    self.configure(height=600)
    self.pack(fill=fill, side=side, expand=expand)


  def refresh(self):
    """Fill in combo boxes"""
    # TODO
    pass


  def update_combo(self):
    """This method must be called right after buffers are cleared"""
    assert self._groups is None
    active_value_dict = {}
    group_keys = tuple(self.groups.keys())
    assert len(group_keys) > 0
    for i, (name, _) in enumerate(group_keys[0]):
      active_value_dict[name] = list(set([gk[i][1] for gk in group_keys]))
    for name, values in active_value_dict.items():
      control = self.active_dict[name]
      assert isinstance(control, ConfigControl)
      control.reset_combo(values)


  # @refresh_friends_at_last
  def set_note(self, note):
    # Revert color of each name label in inactive configs
    if note is None:
      for widget in self.inactive_dict.values():
        widget.indicate_possession(True)
      return

    # Get explicit config control
    hyper_widgets = [
      widget for widget in self.active_dict.values()
      if not widget.is_common and widget.name in self.context.active_flag_set]

    # Set value for each combobox
    need_to_refresh_criteria_panel = False
    for widget in hyper_widgets:
      assert isinstance(widget, ConfigControl)
      if widget.set_value(note.configs[widget.name]):
        need_to_refresh_criteria_panel = True
    if need_to_refresh_criteria_panel:
      self.criteria_panel.refresh()

    # Set value for each inactive combobox
    for widget in self.inactive_dict.values():
      possess = widget.name in note.configs.keys()
      if possess and not widget.is_common:
        widget.set_value(note.configs[widget.name])
      widget.indicate_possession(possess)

  def clear_buffer(self):
    self._candidates = None
    self._groups = None
    self._sorted_hyper = None

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _filter(notes, configs):
    assert isinstance(configs, dict)
    results = []
    for note in notes:
      select = True
      for k, v in configs.items():
        if note.configs[k] != v:
          select = False
          break
      if select: results.append(note)
    return results

  def _get_config_tuple(self, note=None):
    config_tuple = []
    for key in self.sorted_hyper_list:
      value = (note.configs[key] if note is not None
               else self.active_dict[key].current_value)
      config_tuple.append((key, value))
    return tuple(config_tuple)

  # endregion : Private Methods

