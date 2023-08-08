from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font as tkFont

import numpy as np

from tframe.utils.viewer_base.main_frame import Viewer
from tframe.utils.summary_viewer.context import Context


class BaseControl(ttk.Frame):

  WidgetNames = Viewer.WidgetNames
  ruler = None

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
  def to_str(v): return str(v) if type(v) == int else '{:.4f}'.format(v)[:5]


  @staticmethod
  def measure(txt):
    a, b, s = 1.5, 1, 12
    try:
      # TODO: raise `can't invoke "font" command: application has been
      #       destroyed` exception if reopen in the same thread
      if BaseControl.ruler is None:
        BaseControl.ruler = tkFont.Font(family='Helvetica', size=s)
      x = BaseControl.ruler.measure(txt)
    except:
      x = len(txt) * 8
    return int(1.0 * x / s * a + b)
