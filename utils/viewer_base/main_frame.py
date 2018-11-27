from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import tkinter as tk
  import tkinter.ttk as ttk
except Exception as e:
  print(' ! {}'.format(e))
  print(' ! Cannot import tkinter')


DELAY_TO_SHOW_MS = 20

class Viewer(ttk.Frame):
  """Base class for almost all Viewers in tframe"""

  class WidgetNames(object):
    TLabel = 'TLabel'
    TFrame = 'TFrame'
    TLabelframe = 'TLabelframe'
    TButton = 'TButton'
    TRadioButton = 'TRadioButton'
    TCombobox = 'TCombobox'

  def __init__(self, master=None):
    # If root is not provided, load a default one
    if master is None: master = tk.Tk()
    # Call parent's constructor
    ttk.Frame.__init__(self, master)

    # Public attributes
    self.form = master
    self.style = ttk.Style()

    # Bind default key events
    self.form.bind('<Escape>', lambda e: self.form.quit())

  # region : Public Methods

  def set_style(self, *layers, reverse=True, **kwargs):
    assert len(layers) > 0
    assert layers[0] in [s for s in dir(self.WidgetNames) if s[0] == 'T']

    if reverse: layers = reversed(layers)
    style_name = '.'.join(layers)
    self.style.configure(style_name, **kwargs)
    return style_name

  def show(self):
    self.form.after(DELAY_TO_SHOW_MS, self._move_to_center)
    self.form.mainloop()

  # endregion : Public Methods

  # region : Private Methods

  def _move_to_center(self):
    width = self.master.winfo_width()
    height = self.master.winfo_height()
    width_inc = int((self.master.winfo_screenwidth() - width) / 2)
    height_inc = int((self.master.winfo_screenheight() - height) / 2)
    self.master.geometry(
      '{}x{}+{}+{}'.format(width, height, width_inc, height_inc))

  # endregion : Private Methods


if __name__ == '__main__':
  from tframe.utils.viewer_base import main_frame

  viewer = main_frame.Viewer()
  viewer.show()




