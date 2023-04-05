from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
from collections import OrderedDict

import matplotlib
if matplotlib.get_backend() != 'module://backend_interagg':
  matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tkinter as tk
from tkinter import ttk
from tkinter import Frame

from tframe import console
from tframe.utils.tensor_viewer.plugin import Plugin, VariableWithView


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
    self.show_absolute_value = False  # key_symbol: a
    self.show_value = False           # key_symbol: v
    self.unify_range = False          # key_symbol: c
    self.log_scale = True             # key_symbol: g
    self.is_on = True

    self.for_export = False           # key_symbol: e

    self.force_real_shape = False     # key_symbol: E

  @property
  def index(self):
    return (0 if self.related_criteria_figure is None else
            self.related_criteria_figure.cursor)

  # region : Public Methods

  def next_or_previous(self, step, level):
    if not self.is_on: return
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

  def set_variable_dict(self, v_dict, plugins=None):
    """
    Set variable dict to this widgets
    :param v_dict: a dict whose values are lists of numpy arrays
    """
    # Sanity check
    assert isinstance(v_dict, OrderedDict) and len(v_dict) > 0
    # Set variable dict
    def recursively_flatten(src):
      assert isinstance(src, OrderedDict)
      dst = OrderedDict()
      for k, v in src.items():
        if isinstance(v, dict):
          # TODO: a temporal fix for nan situation
          vd = recursively_flatten(v)
          if len(vd) > 0: dst[k] = vd
          else:
            print(' ! Failed to set `{}` to viewer since its empty.'.format(k))
        elif isinstance(v, list):
          # flattened = self._flatten(v, name=k)
          flattened = v
          # Loosely check nan
          if flattened and np.isnan(flattened[0][0]).any():
            print(' ! Failed to set `{}` to viewer. np.nan detected.'.format(k))
            continue
          if flattened is not None: dst[k] = flattened
          else: print(' ! Failed to set `{}` to viewer.'.format(k))
        else: raise TypeError(
            '!! Unknown type {} found in variable dict'.format(type(v)))
      return dst

    # Re-arrange image stack if necessary
    self._variable_dict = recursively_flatten(v_dict)
    # Active plugin
    if plugins:
      assert isinstance(plugins, list)
      for p in plugins:
        assert isinstance(p, Plugin)
        p.modify_variable_dict(self._variable_dict)
    # Initialize combo_boxes according to self._variable_dict
    self._init_combo_boxes()
    # Refresh
    self.refresh()

  def refresh(self):
    if self._variable_dict is None or not self.is_on: return

    # Clear axes (important!! otherwise the memory will not be released)
    self.subplot.cla()
    self.ax2.cla()

    # Get variable
    target = self._variable_dict
    key = None
    for combo in self.combo_boxes:
      assert isinstance(combo, ttk.Combobox)
      key = combo.get()
      target = target[key]
    assert isinstance(target, (tuple, list, VariableWithView))

    # Show target
    # Remove color bar if necessary
    if self._color_bar is not None: self._color_bar.remove()
    self._color_bar = None

    if isinstance(target, VariableWithView): target.display(self)
    else:
      variable = np.squeeze(target[self.index])
      if len(variable.shape) == 1: self._plot_array(variable, target)
      else: self._show_image(variable, target, key)

    # Tight layout
    # TODO: This line may cause the overlap of colorbar and the corresponding
    #  images
    self.figure.tight_layout()

    # Draw update on canvas
    self.figure_canvas.draw()

  def save_as_eps(self, file_name):
    assert isinstance(file_name, str)
    # self.figure.savefig(file_name, format='eps', dpi=1000)
    self.figure.savefig(file_name, format='eps')
    print('>> Figure saved to `{}`'.format(file_name))

  # endregion : Public Methods

  # region : Private Methods

  def set_ax2_invisible(self):
    self.ax2.set_axis_off()

  def _show_image(self, image, images, key):
    self.set_ax2_invisible()
    plt.setp(self.subplot, xticks=[], yticks=[])

    abs_variable = np.abs(image)
    image = abs_variable if self.show_absolute_value else image
    if self.for_export: image = np.transpose(image)

    gate_like = any([re.match(r'\w+_gate', key),
                     re.match(r'.*[Aa]ngle', key),
                     key == 'recurrent_z'])
    # Show heat_map (original cmap for abs is `OrRd`)
    cmap = 'gist_earth' if self.show_absolute_value or gate_like else 'bwr'
    im = self._heat_map(image, cmap=cmap)
    if self.show_value: self._annotate_heat_map(im, image)
    pool = np.abs(images) if self.unify_range else abs_variable
    # Set color limits
    if gate_like:
      im.set_clim(0, 1)
    elif self.show_absolute_value:
      im.set_clim(np.min(pool), np.max(pool))
    else:
      lim = np.max(pool)
      im.set_clim(-lim, lim)

    title = '|T|' if self.show_absolute_value else 'T'
    title += '({}x{})'.format(image.shape[0], image.shape[1])
    title += ', min={:.2f}, max={:.2f}'.format(np.min(image), np.max(image))
    if not self.for_export:
      self.subplot.set_title(title)
      # self.subplot.set_aspect('equal', adjustable='datalim', anchor='C')
      self.subplot.set_aspect('auto', anchor='C')
    if self.for_export:
      self.subplot.set_aspect('auto')
      # self.subplot.set_xlabel('time step')
    # TODO: beta
    if self.force_real_shape:
      self.subplot.set_aspect(image.shape[0] / image.shape[1])

  def _plot_array(self, array, arrays):
    self.set_ax2_invisible()

    step =  np.arange(len(array)) + 1
    self.subplot.plot(step, array)
    self.subplot.set_xlim(min(step), max(step))
    # TODO: `use_clim` is not appropriate here
    pool = arrays if self.unify_range else array
    self.subplot.set_ylim(np.min(pool), np.max(pool))
    self.subplot.set_aspect('auto')

    self.subplot.grid(True)
    self.subplot.set_yscale('log' if self.log_scale else 'linear')
    if self.log_scale:
      self.subplot.set_ylim(max(np.min(pool), 1e-17), np.max(pool))
    # self.subplot.set_title('Title')
    self.subplot.set_title('{:.3f}-{:.3f}'.format(min(array), max(array)))

  def _create_layout(self):
    # Create figure canvas
    self.figure = plt.Figure()
    self.figure.set_facecolor('white')
    self.subplot = self.figure.add_subplot(111, autoscale_on=True)
    self.ax2 = self.subplot.twinx()
    # pyplt.sca(self.subplot)
    # ... (modify style)
    self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
    try: self.figure_canvas.show()
    except: print(' ! self.figure_canvas.show() failed')
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
    assert isinstance(variable, np.ndarray)
    # console.split()
    # console.show_status('shape = {}'.format(variable.shape))
    # console.split()
    if len(variable.shape) == 3:
      im = self.subplot.imshow(variable, **kwargs)
    else:
      im = self.subplot.imshow(variable, **kwargs)
      # Create color bar
      # if self._color_bar is not None: self._color_bar.remove()
      # self._color_bar = self.figure.colorbar(im, ax=self.subplot)
      divider = make_axes_locatable(self.subplot)
      cax2 = divider.append_axes('right', size='5%', pad=0.05)
      self._color_bar = self.figure.colorbar(im, cax=cax2)

    return im

  def _annotate_heat_map(self, im, data, valfmt='{x:.2f}',
                         text_colors=('black', 'white'), threshold=None):
    # Sanity check
    assert isinstance(data, np.ndarray)
    im_data = im.get_array()
    for d in np.shape(im_data):
      if d > 50: return

    # Normalize the threshold to the images color range.
    if threshold is not None:
      threshold = im.norm(threshold)
    else:
      # threshold = im.norm(im_data.max()) / 2.
      threshold = np.max(np.abs(im_data)) / 2.

    # Set alignment to center
    kw = dict(horizontalalignment='center', verticalalignment='center')

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
      valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a 'Text' for each 'pixel'
    # Change the text's color depending on the data
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        val = im_data[i, j]
        if not self.show_absolute_value:
          val = np.abs(val)
        kw.update(color=text_colors[val > threshold])
        im.axes.text(j, i, valfmt(data[i, j], None), **kw)

  # endregion : Private Methods

  # region : Utils

  @staticmethod
  def _flatten(tensor_list, name):
    """Try to re-arrange an image stack, say, of shape (h, w, N) into
        a single image of shape (H, W)
    """
    assert isinstance(tensor_list, list)
    tensor = tensor_list[0]
    if len(tensor.shape) in (2, 1): return tensor_list
    elif len(tensor.shape) == 3 and tensor.shape[2] == 3: return tensor_list
    elif len(tensor.shape) == 3 and tensor.shape[2] == 1:
      return [t.reshape(t.shape[:2]) for t in tensor_list]
    elif len(tensor.shape) == 4 and tensor.shape[2] != 3: return None
    elif len(tensor.shape) not in (3, 4): return None
    # Now len(tensor.shape) in (3, 4)
    console.show_status(
      'Converting `{}` with shape {} ...'.format(name, tensor.shape))
    h, w = tensor.shape[:2]
    total = tensor.shape[-1]
    edge = int(np.ceil(np.sqrt(total)))
    H, W = h * edge + edge - 1, w * edge + edge - 1
    new_shape = [H, W] + [3] if len(tensor.shape) == 4 else []
    new_list = []
    for t in tensor_list:
      max_value = np.max(t)
      pie = np.zeros(shape=new_shape, dtype=np.float32)
      for i in range(total):
        I = i // edge
        J = i - I * edge
        h_from = I * h + I
        i_slice = slice(h_from, h_from + h)
        w_from = J * w + J
        j_slice = slice(w_from, w_from + w)
        if len(tensor.shape) == 3:
          pie[i_slice, j_slice] = t[:, :, i] / max_value
        else: pie[i_slice, j_slice, :] = t[:, :, :, i] / max_value
      new_list.append(pie)
      # console.print_progress(i, total)
    return new_list

  # endregion : Utils


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
