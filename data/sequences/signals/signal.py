from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import checker


class Signal(np.ndarray):
  """Base class of all signals. If saving and loading is needed, put a
     Signal into a container which maintains its additional attributes"""

  def __new__(cls, input_array, fs):
    if not isinstance(input_array, np.ndarray):
      raise TypeError("!! Input array for constructing signal should be an "
                      "instance of numpy.ndarray")
    obj = input_array.view(cls)
    obj.fs = fs
    return obj

  # region : Properties

  # region : Additional Attributes

  @property
  def fs(self):
    assert hasattr(self, '_fs')
    assert isinstance(self._fs, (int, float)) and self._fs > 0
    return self._fs

  @fs.setter
  def fs(self, val):
    assert isinstance(val, (int, float)) and val > 0
    self._fs = val

  # endregion : Additional Attributes

  # region : Basic Properties

  @property
  def is_real(self):
    return not np.iscomplex(self).any()

  @property
  def duration(self):
    return self.size / self.fs

  @property
  def norm(self):
    return float(np.linalg.norm(self))

  @property
  def rms(self):
    return float(np.sqrt(np.mean(np.square(self))))

  # endregion : Basic Properties

  # region : Properties in Transfer Domain

  @property
  def spectrum(self):
    spectrum = np.fft.fft(self)
    # return spectrum
    if not self.is_real:
      return np.fft.fftshift(spectrum)
    else:
      freqs = np.fft.fftfreq(self.size, 1. / self.fs)
      return spectrum[freqs >= 0]

  @property
  def amplitude_spectrum(self):
    # Here we apply a scaling factor of 1/fs so that the amplitude of the
    #  FFT at a frequency component equals that of the CFT and to preserve
    #  Parseval's theorem
    return np.abs(self.spectrum) / self.fs

  @property
  def power_spectrum_density(self):
    return np.abs(self.spectrum * np.conj(self.spectrum) / self.size / self.fs)

  # endregion : Properties in Transfer Domian

  # region : Statistic Properties

  @property
  def average(self):
    return float(np.average(self))

  @property
  def variance(self):
    return float(np.var(self))

  # endregion : Statistic Properties

  # region : Properties for Plotting

  @property
  def time_axis(self):
    return np.linspace(0, 1, self.size) * self.duration

  @property
  def freq_axis(self):
    freqs = np.fft.fftfreq(self.size, 1 / self.fs)
    if self.is_real: return freqs[freqs >= 0]
    else: return np.fft.fftshift(freqs)

  # endregion : Properties for Plotting

  # endregion : Properties

  # region : Public Methods

  def causal_matrix(self, memory_depth, skip_head=False):
    checker.check_positive_integer(memory_depth)
    assert isinstance(self, np.ndarray)
    if memory_depth == 1: return np.reshape(self, (-1, 1))
    N, D = self.size, memory_depth
    x = np.append(np.zeros(shape=(D - 1,)), self)
    matrix = np.zeros(shape=(N, D))
    for i in range(N): matrix[i] = x[i:i+D]
    return matrix[D - 1:] if skip_head else matrix

  def auto_correlation(self, lags, keep_dim=False):
    if isinstance(lags, int):
      lags = (0, lags)
    if not (isinstance(lags, tuple) or isinstance(lags, list)):
      raise TypeError('!! Input lags must be a a tuple or a list')
    results = np.ones_like(self)
    for lag in lags:
      results *= np.append(np.zeros((lag,)), self)[:self.size]

    if keep_dim: return results
    return float(np.average(results))

  def plot(self, form_title='Untitled', show_time_domain=False,
           show_freq_domain=True, db=True):
    from tframe.data.sequences.signals.figure import Figure, Subplot
    fig = Figure(form_title)
    if show_time_domain: fig.add(Subplot.TimeDomainPlot(self))
    if show_freq_domain:
      fig.add(Subplot.AmplitudeSpectrum(self, db=db))
    fig.plot()

  # endregion : Public Methods

  # region : Superclass Preserved

  def __array_finalize__(self, obj):
    if isinstance(obj, Signal): self.fs = obj.fs

  # endregion : Superclass Preserved

  # region : Static Methods

  @staticmethod
  def gaussian_white_noise(intensity, size, fs):
    """Generate a gaussian white noise with specific intensity A.
       That is, R_XX(\tau) = A\delta_(\tau), S_XX(\omega) = A
       R_XX(\tau) = E[x[t]x[t-\tau]] = \sigma^2 \delta(\tau)
       Reference: https://www.gaussianwaves.com/2013/11/simulation-and-analysis-of-white-noise-in-matlab/"""
    noise = np.random.normal(scale=np.sqrt(intensity), size=size)
    return Signal(noise, fs=fs)

  @staticmethod
  def multi_tone(freqs, fs, duration, vrms=None, phases=None,
                 noise_power=0):
    """Generate multi-tone signal.
        numpy will perfectly handle the situation when 1/fs is not an integer"""
    # Determine the root-mean-square voltage
    if vrms is None: vrms = np.ones_like(freqs)
    if len(vrms) != len(freqs):
      raise ValueError('Length of freqs must be the same as vrms')
    # Determine phases
    if phases is None: phases = np.zeros_like(freqs)
    if len(phases) != len(freqs):
      raise ValueError('Length of freqs must be the same as phases')
    t = np.arange(0, duration, 1 / fs)
    x = np.zeros_like(t)
    for i in range(len(freqs)):
      x += vrms[i] * np.sqrt(2) * np.cos(2 * np.pi * freqs[i] * t + phases[i])
    # Instantiate Signal
    signl = Signal(x, fs=fs)
    # Add gaussian white noise to signal
    # Reference: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html
    noise_power = noise_power * signl.fs / 2
    noise = Signal.gaussian_white_noise(
      noise_power, size=signl.shape, fs=signl.fs)
    return signl + noise

  # endregion : Static Methods


if __name__ == '__main__':
  pass



