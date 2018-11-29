from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk

from .context import Context

from tframe.utils.viewer_base.main_frame import Viewer


class BaseControl(ttk.Frame):

  WidgetNames = Viewer.WidgetNames

  def __init__(self, master):

    # Call parent's constructor
    ttk.Frame.__init__(self, master)

    # Attributes
    context = getattr(master, 'context', None)
    while context is None:
      master = getattr(master, 'master', None)
      if master is None: break
      else: context = getattr(master, 'context', None)
    assert isinstance(context, Context)
    self.context = context

    assert hasattr(master, 'set_style')
    self._set_style = getattr(master, 'set_style')


  def load_to_master(self, side, fill=tk.BOTH, expand=True):
    raise NotImplementedError


  def refresh(self, *args, **kwargs):
    raise NotImplementedError


  def set_style(self, *layers, reverse=True, **kwargs):
    return self._set_style(*layers, reverse=reverse, **kwargs)


  @staticmethod
  def to_str(v):
    return str(v) if v == int(v) else '{:.2f}'.format(v)
