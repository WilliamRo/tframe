from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe.data.sequences.signals.tf_signal import Signal

try:
  from matplotlib import pyplot as plt
except:
  print('!! Failed to import matplotlib, class Figure disabled')


class Figure(object):
  """Class for plot signals in different domains"""
  def __init__(self, figtitle="Untitled", figsize=(10, 8), subplots=None):
    self.figure = plt.figure(figtitle, figsize=figsize, facecolor='w')
    self.figure.canvas.mpl_connect('key_press_event', Figure._on_key)
    self.subplots = []

    if subplots is not None and (
          isinstance(subplots, list) or isinstance(subplots, tuple)):
      for subplot in subplots: self.add(subplot)

  # region : Public Methods

  def plot(self, show=True, ylim=None):
    nrows = len(self.subplots)
    if nrows == 0:
      print('>> Nothing to be plotted')
      return

    # Plot
    row = 0
    for subplot in self.subplots:
      assert isinstance(subplot, Subplot)
      row += 1
      plt.subplot(nrows, 1, row)
      subplot.plot()
      if ylim is not None:
        if not isinstance(ylim, (list, tuple)) and ylim is True:
          ylim = plt.ylim()
        else: plt.ylim(ylim)

    # Adjust subplot layout
    plt.subplots_adjust(hspace=0.4, left=0.1, right=0.92,
                        bottom=0.07, top=0.95)
    plt.tight_layout(pad=1.6)

    # Show figure
    if show: plt.show()

    return self.figure

  def add(self, subplot):
    if not isinstance(subplot, Subplot):
      raise TypeError('Input must be an instance of Subplot')
    self.subplots.append(subplot)

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _on_key(event):
    if event.key in ['escape']:
      plt.close()

  # endregion : Private Methods

  """For some reason, do not delete this line"""


class Subplot(object):
  """Subplot definition in a figure"""

  def __init__(self, signl, plot, db=False, prefix=None, **kwargs):
    if signl is not None: assert isinstance(signl, Signal)
    assert callable(plot)
    self.signl = signl
    self.plot = lambda: plot(self)
    self.db = db
    self.predix = prefix
    self.kwargs = kwargs

  # region : Properties

  @property
  def ys_and_legends(self):
    ys = [self.signl]
    legends = ['Signal']
    for key, val in self.kwargs.items():
      if isinstance(val, Signal) and val.size == self.signl.size:
        ys.append(val)
        legends.append(key)
    return ys, legends

  # endregion : Properties

  # region : Static Methods

  @staticmethod
  def TimeDomainPlot(signl, prefix=None, **kwargs):
    return Subplot(
      signl, Subplot._time_domain_plot, db=False, prefix=prefix, **kwargs)

  @staticmethod
  def AmplitudeSpectrum(signl, db=True, prefix=None, **kwargs):
    return Subplot(
      signl, Subplot._amplitude_spectrum, db=db, prefix=prefix, **kwargs)

  @staticmethod
  def PowerSpectrum(signl, db=True, prefix=None, **kwargs):
    return Subplot(
      signl, Subplot._power_spectrum, db=db, prefix=prefix, **kwargs)

  @staticmethod
  def Default(x, ys, title=None, prefix=None,
                xlabel=None, ylabel=None, legends=None):
    return Subplot(None, Subplot._default_plot, prefix=prefix, x=x, ys=ys,
                    xlabel=xlabel, ylabel=ylabel, legends=legends, title=title)

  @staticmethod
  def Custom(plot_function):
    return Subplot(None, plot_function)

  # endregion : Static Methods

  # region : Private Methods

  def _default_plot(self):
    x = self.kwargs.get('x', None)
    ys = self.kwargs.get('ys', None)
    if x is None or ys is None:
      raise ValueError('!! x and ys must not be None')
    title = self.kwargs.get('title', None)
    xlabel = self.kwargs.get('xlabel', None)
    ylabel = self.kwargs.get('ylabel', None)
    legends = self.kwargs.get('legends', None)

    Subplot._plot(x, ys, title=title, xlabel=xlabel, ylabel=ylabel,
                  legends=legends)

  def _time_domain_plot(self):
    xlabel = 'Time'
    if self.signl.fs is not None: xlabel += ' (s)'
    ys, legends = self.ys_and_legends
    Subplot._plot(self.signl.time_axis, ys, legends=legends,
                  title=self._make_title('Time Domain Plot'),
                  xlabel=xlabel, ylabel='Voltage')

  def _amplitude_spectrum(self):
    ys, legends = self.ys_and_legends
    densities = []
    for y in ys:
      assert isinstance(y, Signal)
      densities.append(20 * np.log10(y.amplitude_spectrum + 1e-12)
                       if self.db else y.amplitude_spectrum)
    # Subplot._plot(self.signl.freq_axis, density)
    ylabel = 'Spectrum Amplitude'
    if self.db: ylabel += ' (dB)'
    Subplot._plot(self.signl.freq_axis, densities, legends=legends,
                  title=self._make_title('Amplitude Spectrum'),
                  xlabel='Frequency (Hz)', ylabel=ylabel)

  def _power_spectrum(self):
    ys, legends = self.ys_and_legends
    densities = []
    for y in ys:
      assert isinstance(y, Signal)
      densities.append(10 * np.log10(y.power_spectrum_density + 1e-12)
                       if self.db else y.power_spectrum_density)
    # Subplot._plot(self.signl.freq_axis, density)
    ylabel = 'Power'
    if self.db: ylabel += ' (dB)'
    Subplot._plot(self.signl.freq_axis, densities, legends=legends,
                  title=self._make_title('Power Spectrum'),
                  xlabel='Frequency (Hz)', ylabel=ylabel)

  def _make_title(self, title):
    if self.predix is not None: title = '[{}] {}'.format(self.predix, title)
    return title

  @staticmethod
  def _plot(x, ys, title=None, xlabel=None, ylabel=None, grid=True,
            **kwargs):
    # Check inputs
    legends = kwargs.get('legends', None)
    if not isinstance(ys, list) and not isinstance(ys, tuple): ys = (ys,)
    if x is None: x = np.arange(ys[0].size)

    args = ()
    for y in ys:
      assert y is not None and isinstance(y, np.ndarray)
      assert len(y.shape) == 1
      args += (x, y)

    plt.plot(*args)
    plt.xlim(x[0], x[-1])

    if title is not None:
      plt.title(title)
    if xlabel is not None:
      plt.xlabel(xlabel)
    if ylabel is not None:
      plt.ylabel(ylabel)
    if legends is not None:
      plt.legend(legends)

    plt.grid(grid)

  # endregion : Private Methods

  """For some reason, do not delete this line"""


if __name__ == '__main__':
  # Define signal
  fs = 2000
  duration = 1
  freqs = [500, 800]
  vrms = [2, 1]
  phases = [0, np.pi]
  signal = Signal.multi_tone(
    freqs, fs, duration, vrms=vrms, phases=phases, noise_power=1e-2)

  # Plot signal
  fig = Figure()
  fig.add(Subplot.TimeDomainPlot(signal, prefix='Input signal'))
  fig.add(Subplot.AmplitudeSpectrum(signal, db=True))
  fig.plot()

