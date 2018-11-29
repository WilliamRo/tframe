from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .base_control import BaseControl


# region : Decorators

def refresh_friends_at_last(method):
  def wrapper(self, *args, **kwargs):
    assert isinstance(self, (ConfigControl, ConfigPanel))
    method(self, *args, **kwargs)
    self.header.refresh()
    self.criteria_panel.refresh()
  return wrapper

# endregion : Decorators


class ConfigControl(BaseControl):

  # Basic configurations
  WIDTH = 400
  HEIGHT = 100
  MAX_STR_LENGTH = 50

  def __init__(self, master, flag_name, flag_values, is_active):
    # Sanity check
    assert isinstance(flag_name, str) and isinstance(flag_values, (tuple, list))
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Attributes
    self.name = flag_name
    assert len(flag_values) > 0
    self.values = flag_values
    assert isinstance(is_active, bool)
    self._active = is_active

    # Ancestors and friends
    self.config_panel = self.master.master
    assert isinstance(self.config_panel, ConfigPanel)
    self.main_frame = self.config_panel.main_frame
    self.header = self.main_frame.header
    self.criteria_panel = self.main_frame.criteria_panel

    # Layout
    self.switch_button = ttk.Button(self, cursor='hand2')
    self.label_name = ttk.Label(self)
    self.values_control = None
    self._create_layout()

  # region : Properties

  @property
  def is_common(self):
    return len(self.values) == 1

  @property
  def current_value(self):
    if self.is_common:
      return self.values[0]
    else:
      assert isinstance(self.values_control, ttk.Combobox)
      return self.values[self.values_control.current()]

  # endregion : Properties

  # region : Public Methods

  def load_to_master(self, side=tk.TOP, fill=tk.X, expand=False):
    self.pack(side=side, fill=fill, expand=expand)

  def refresh(self):
    pass

  def set_value(self, val):
    index = self.values.index(val)
    assert index >= 0 and isinstance(self.values_control, ttk.Combobox)
    self.values_control.current(index)

  # endregion : Public Methods

  # region : Events

  @refresh_friends_at_last
  def _on_combobox_selected(self, _):
    # TODO:
    if not self._active: return

  @refresh_friends_at_last
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

    # (3) Value
    if self.is_common:
      self.values_control = ttk.Label(self, text=str(self.values[0]))
    else:
      self.values_control = ttk.Combobox(
        self, state='readonly', justify=tk.RIGHT)
      self.values_control.config(values=self.values)
      self.values_control.current(0)
      self.values_control.bind(
        '<<ComboboxSelected>>', self._on_combobox_selected)
    self.values_control.pack(side=tk.RIGHT)

  # endregion : Private Methods


class ConfigPanel(BaseControl):

  def __init__(self, master):
    # Call parent's constructor
    BaseControl.__init__(self, master)

    # Widgets
    self.hyper_parameters = ttk.LabelFrame(self, text='Hyper-Parameters')
    self.common_configs = ttk.LabelFrame(self, text='Common Configurations')
    self.inactive_configs = ttk.LabelFrame(self, text='Inactive Configurations')

    # Attributes
    self.active_dict = {}
    self.inactive_dict = {}

    # Ancestor and friends
    self.main_frame = self.master.master

  # region : Properties

  @property
  def criteria_panel(self):
    return self.main_frame.criteria_panel

  @property
  def header(self):
    return self.main_frame.header

  @property
  def active_config_dict(self):
    return {k: self.active_dict[k].current_value
            for k in self.context.active_flag_set}
  
  @property
  def qualified_notes(self):
    flag_of_interest = set(self.active_config_dict.keys())
    return [note for note in self.context.notes
            if set(note.configs.keys()).issuperset(flag_of_interest)]

  @property
  def selected_notes(self):
    notes = []
    config_dict = self.active_config_dict

    for note in self.qualified_notes:
      select = True
      for k, v in config_dict.items():
        if note.configs[k] != v:
          select = False
          break
      if select: notes.append(note)

    return notes

  @property
  def minimum_height(self):
    h_empty_panel = 21
    h_each_control = 27
    coef = 3 if len(self.active_dict) == 0 else 2
    return 3 * h_empty_panel + (coef + len(self.active_dict)) * h_each_control

  # endregion : Properties

  # region : Public Methods

  def initialize_config_controls(self):
    # self.inactive_dict, self.active_dict = {}, {}
    for k, v in self.context.flag_value_dict.items():
      # Create an active one
      master = self.hyper_parameters if len(v) > 1 else self.common_configs
      active_control = ConfigControl(master, k, v, True)
      self.active_dict[k] = active_control
      # Create an inactive one
      inactive_control = ConfigControl(self.inactive_configs, k, v, False)
      self.inactive_dict[k] = inactive_control

    # Clear 3 panels TODO: reloading is not allowed
    # self.hyper_parameters.

    # Pack config controls
    for k in self.context.active_flag_set:
      self.active_dict[k].load_to_master()
    for k in self.context.inactive_flag_set:
      self.inactive_dict[k].load_to_master()


  def load_to_master(self, side=tk.LEFT, fill=tk.Y, expand=True):
    # Pack label-frames
    pack_params = {'fill': tk.BOTH, 'side': tk.TOP, 'expand': False}
    for label_frame in (
        self.hyper_parameters,
        self.common_configs,
        self.inactive_configs,
    ):
      label_frame.configure(width=400, height=48)
      label_frame.pack(**pack_params)
    self.inactive_configs.pack(expand=True)
    self.inactive_configs.configure()

    # Pack self
    self.configure(height=600)
    self.pack(fill=fill, side=side, expand=expand)


  def refresh(self):
    pass


  @refresh_friends_at_last
  def set_note(self, note):
    # Get explicit config control
    config_controls = [
      control for control in self.active_dict.values()
      if not control.is_common and control.name in self.context.active_flag_set]

    # Set value for each combobox
    for control in config_controls:
      assert isinstance(control, ConfigControl)
      control.set_value(note.configs[control.name])

  # endregion : Public Methods

  # region : Private Methods



  # endregion : Private Methods

