from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from tkinter import Frame

if matplotlib.get_backend() != 'module://backend_interagg':
  matplotlib.use("TkAgg")


class LossFigure(Frame):
  # Configurations of LossFigure widget
  WIDTH = 500
  HEIGHT = 400
  PCT_PER_STEP = 0.05
  SLIDER_WIDTH = 0.05

  def __init__(self, master):
    # Call parent's constructor
    Frame.__init__(self, master)

    # Layout
    self.figure = None
    self.subplot = None
    self.figure_canvas = None
    self.tk_canvas = None
    self.scroll_bar = None
    self._create_layout()

    # Attributes
    self._step = None
    self._loss = None
    self._index = None

    self.related_variable_viewer = None

  # region : Properties

  @property
  def index(self):
    return 0 if self._index is None else self._index

  # endregion : Properties

  # region : Public Methods

  def set_step_and_loss(self, step, loss):
    # Sanity check
    if isinstance(step, (list, tuple)):
      step = np.array(step)
    assert isinstance(step, np.ndarray) and isinstance(loss, np.ndarray)
    assert len(loss.shape) == len(step.shape) == 1 and loss.size == step.size

    # Set step and loss
    self._step = step
    self._loss = loss
    self._index = 0

    # Refresh figure
    self.refresh()

  def refresh(self):
    if self._step is None or self._loss is None:
      return
    # Set slider bar
    self._set_slider()
    # Clear
    self.subplot.cla()
    self.subplot.set_xlabel('Epoch')
    self.subplot.set_ylabel('Loss')
    # Plot loss curve
    self.subplot.plot(self._step, self._loss)
    # Plot current loss mark
    assert 0 <= self._index < len(self._step)
    step, loss = self._step[self._index], self._loss[self._index]
    self.subplot.plot(step, loss, 'rs')
    # Refresh title
    title = 'Loss = {:.3f}'.format(loss)
    self.subplot.set_title(title)
    # Tight layout
    self.figure.tight_layout()
    # Draw update on canvas
    self.figure_canvas.draw()

    # Refresh related variable viewer if necessary
    if self.related_variable_viewer is not None:
      self.related_variable_viewer.refresh()

  def on_scroll(self, action, *args):
    if self._step is None or self._loss is None:
      return

    moveto = 'moveto'
    scroll = 'scroll'
    assert action in (moveto, scroll)

    # Set cursor corresponding to action
    if action == moveto:
      # offset \in [0.0, 1.0 - SLIDER_WIDTH]
      offset = float(args[0]) / (1.0 - self.SLIDER_WIDTH)
      offset = min(offset, 1.0)
      self._index = int(np.round(offset * (len(self._step) - 1)))
    elif action == scroll:
      # step \in {-1, 1}
      step, what = args
      step = int(step)
      index = self._index + np.round(
        step * self.PCT_PER_STEP * len(self._step))
      index = min(max(0, index), len(self._step) - 1)
      if index == self._index: return
      else: self._index = int(index)

    # Refresh subplot
    self.refresh()


  # endregion : Public Methods

  # region : Private Methods

  def _create_layout(self):
    # Create figure canvas
    self.figure = plt.Figure()
    self.figure.set_facecolor('white')
    self.subplot = self.figure.add_subplot(111)
    self.subplot.set_xlabel('Epoch')
    self.subplot.set_ylabel('Loss')
    self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
    self.figure_canvas.show()
    self.tk_canvas = self.figure_canvas.get_tk_widget()
    self.tk_canvas.configure(height=self.HEIGHT, width=self.WIDTH)
    self.tk_canvas.pack(fill=tk.BOTH)

    # Create scroll bar
    scroll_bar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
    scroll_bar.pack(fill=tk.X, side=tk.BOTTOM)
    self.scroll_bar = scroll_bar

    self.scroll_bar.configure(command=self.on_scroll)

  def _set_slider(self):
    """ Set slider position according to index
    """
    # Calculate relative index \in [0.0, 1.0]
    rel_index = 1. * self._index / (len(self._step) - 1)
    rel_index *= (1 - self.SLIDER_WIDTH)

    lo = max(0.0, rel_index)
    hi = min(1.0, rel_index + self.SLIDER_WIDTH)
    self.scroll_bar.set(lo, hi)

  # endregion : Private Methods


if __name__ == '__main__':
  t = np.arange(-2, 2, step=0.01)
  loss = np.sin(t)

  root = tk.Tk()
  # root.geometry('500x600+400+100')
  root.bind('<Escape>', lambda _: root.quit())
  lf = LossFigure(root)
  lf.pack(fill=tk.BOTH)
  # lf.configure(bg='red', height=200)
  # lf.set_step_and_loss(t, loss)
  root.mainloop()


"""
slider offset range: [0, SLIDER_WIDTH]
"""
