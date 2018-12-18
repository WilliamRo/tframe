from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import OrderedDict

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from tkinter import Frame

if matplotlib.get_backend() != 'module://backend_interagg':
  matplotlib.use("TkAgg")


class VariableViewer(Frame):
  # Configurations of VariableViewer widget
  WIDTH = 500
  HEIGHT = 500

  def __init__(self, master):
    # Call parent's constructor
    Frame.__init__(self, master)

    # Attributes
    self._variable_dict = None
    self._variable_names = None
    self.related_criteria_figure = None

    # Layout
    self.figure = None
    self.subplot = None
    self.figure_canvas = None
    self.tk_canvas = None

    self.combo_boxes = []
    self._create_layout()
    self._color_bar = None

    # Options
    self.show_absolute_value = True  # key_symbol: a
    self.show_value = False          # key_symbol: v
    self.use_clim = True             # key_symbol: c

  @property
  def index(self):
    return (0 if self.related_criteria_figure is None else
            self.related_criteria_figure.cursor)

  # region : Public Methods

  def next_or_previous(self, step, level):
    assert step in (1, -1) and level in (0, 1)
    if len(self.combo_boxes) == 1: level = 0
    combo_box = self.combo_boxes[level]
    name_list = self._variable_names[level]
    assert isinstance(name_list, tuple)

    i = name_list.index(combo_box.get()) + step
    if i < 0: i = len(name_list) - 1
    if i >= len(name_list): i = 0
    # Set combo box to next or previous entry
    combo_box.set(name_list[i])
    self.refresh()

  def set_variable_dict(self, v_dict):
    """
    Set variable dict to this widgets
    :param v_dict: a dict whose values are lists of numpy arrays
    """
    # Sanity check
    assert isinstance(v_dict, OrderedDict) and len(v_dict) > 0
    # Set variable dict
    self._variable_dict = v_dict
    self._init_combo_boxes()
    # Refresh
    self.refresh()

  def refresh(self):
    if self._variable_dict is None: return

    # Clear axes (important!! otherwise the memory will not be released)
    self.subplot.cla()
    plt.setp(self.subplot, xticks=[], yticks=[])
    # Get variable
    target = self._variable_dict
    for combo in self.combo_boxes:
      assert isinstance(combo, ttk.Combobox)
      target = target[combo.get()]
    assert isinstance(target, (tuple, list))

    # Show image
    variable = target[self.index]
    image = np.abs(variable) if self.show_absolute_value else variable

    # Show heat_map
    im = self._heat_map(image, cmap='YlGn')
    if self.show_value: self._annotate_heat_map(im, variable)
    if self.use_clim:
      # TODO
      v = target
      if self.show_absolute_value: v = np.abs(v)
      im.set_clim(np.min(v), np.max(v))

    title = '|T|' if self.show_absolute_value else 'T'
    title += '({}x{})'.format(variable.shape[0], variable.shape[1])
    self.subplot.set_title(title)

    # Tight layout
    self.figure.tight_layout()

    # Draw update on canvas
    self.figure_canvas.draw()

  # endregion : Public Methods

  # region : Private Methods

  def _create_layout(self):
    # Create figure canvas
    self.figure = plt.Figure()
    self.figure.set_facecolor('white')
    self.subplot = self.figure.add_subplot(111)
    plt.setp(self.subplot, xticks=[], yticks=[])
    # plt.sca(self.subplot)
    # plt.xticks([1, 2, 3])
    # ... (modify style)
    self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
    self.figure_canvas.show()
    self.tk_canvas = self.figure_canvas.get_tk_widget()
    self.tk_canvas.configure(height=self.HEIGHT, width=self.WIDTH)
    self.tk_canvas.pack(fill=tk.BOTH)

    # Create label
    label = ttk.Label(self, text=' Selected Tensor :  ')
    label.pack(side=tk.LEFT)

  def _init_combo_boxes(self):
    """Currently at most 2 combo boxes are supported"""
    # Clear old combo boxes
    for combo in self.combo_boxes: combo.pack_forget()
    self.combo_boxes = []

    # Get hierarchical tensor names
    assert isinstance(self._variable_dict, OrderedDict)
    self._variable_names = []
    self._variable_names.append(tuple(self._variable_dict.keys()))
    val0 = tuple(self._variable_dict.values())[0]
    while isinstance(val0, OrderedDict):
      self._variable_names.append(tuple(val0.keys()))
      val0 = tuple(val0.values())[0]

    # Create and pack combo boxes
    bind_combo = lambda c: c.bind(
      '<<ComboboxSelected>>', lambda _: self.refresh())
    for keys in self._variable_names:
      combo = ttk.Combobox(self, state='readonly', value=keys)
      combo.pack(side=tk.LEFT, fill=tk.X, expand=1)
      combo.set(keys[0])
      bind_combo(combo)
      self.combo_boxes.append(combo)

  def _heat_map(self, variable, **kwargs):
    """
    ref: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    """
    kwargs['interpolation'] = 'none'

    # Plot the heat map
    im = self.subplot.imshow(variable, **kwargs)

    # Create color bar
    if self._color_bar is not None: self._color_bar.remove()
    self._color_bar = self.figure.colorbar(im, ax=self.subplot)
    return im

  def _annotate_heat_map(self, im, data, valfmt='{x:.2f}',
                         text_colors=('black', 'white'), threshold=None):
    # Sanity check
    assert isinstance(data, np.ndarray)
    im_data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
      threshold = im.norm(threshold)
    else:
      threshold = im.norm(im_data.max()) / 2.

    # Set alignment to center
    kw = dict(horizontalalignment='center', verticalalignment='center')

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
      valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a 'Text' for each 'pixel'
    # Change the text's color depending on the data
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        kw.update(color=text_colors[im.norm(im_data[i, j]) > threshold])
        im.axes.text(j, i, valfmt(data[i, j], None), **kw)

  # endregion : Private Methods


if __name__ == '__main__':
  w1 = np.random.random(size=(5, 5))
  w2 = np.random.random(size=(6, 8))
  w_dict = OrderedDict()
  w_dict['w1'] = [w1]
  w_dict['w2'] = [w2]

  root = tk.Tk()
  root.bind('<Escape>', lambda _: root.quit())
  vv = VariableViewer(root)
  vv.pack(fill=tk.BOTH)
  vv.set_variable_dict(w_dict)
  vv.master.title('Variable Viewer')
  root.mainloop()
