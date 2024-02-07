from roma import console
from roma import Nomear

import numpy as np



class ICC(Nomear):
  class Types:
    ICC1 = 'ICC(1)'
    ICC2A = 'ICC(A, 1)'
    ICC2C = 'ICC(C, 1)'
    ICC3A = 'ICC2A'
    ICC3C = 'ICC2C'

  def __init__(self, data_matrix, icc_type=Types.ICC2A):
    self.data = data_matrix
    self.type = icc_type

  # region: Properties

  @property
  def data(self) -> np.ndarray:
    return self.get_from_pocket('data_matrix', key_should_exist=True)

  @data.setter
  def data(self, data_matrix):
    assert isinstance(data_matrix, np.ndarray)
    assert len(data_matrix.shape) == 2
    self.put_into_pocket('data_matrix', data_matrix)

  @Nomear.property()
  def n(self): return self.data.shape[0]

  @Nomear.property()
  def k(self): return self.data.shape[1]

  @Nomear.property()
  def S(self): return np.mean(self.data, axis=1)

  @Nomear.property()
  def M(self): return np.mean(self.data, axis=0)

  @Nomear.property()
  def x_bar(self): return np.mean(self.S)

  @Nomear.property()
  def SST(self): return np.sum(np.square(self.data - self.x_bar))

  @Nomear.property()
  def SSBS(self): return self.k * np.sum(np.square(self.S - self.x_bar))

  @Nomear.property()
  def SSBM(self): return self.n * np.sum(np.square(self.M - self.x_bar))

  @Nomear.property()
  def SSWS(self): return np.sum(np.square(self.data - self.S.reshape(-1, 1)))

  @Nomear.property()
  def SSWM(self): return np.sum(np.square(self.data - self.M.reshape(1, -1)))

  @Nomear.property()
  def SSE(self): return self.SST - self.SSBS - self.SSBM

  @Nomear.property()
  def MST(self): return self.SST / (self.n * self.k - 1)

  @Nomear.property()
  def MSBS(self): return self.SSBS / (self.n - 1)

  @Nomear.property()
  def MSBM(self): return self.SSBM / (self.k - 1)

  @Nomear.property()
  def MSWS(self): return self.SSWS / (self.n * (self.k - 1))

  @Nomear.property()
  def MSWM(self): return self.SSWM / ((self.n - 1) * self.k)

  @Nomear.property()
  def MSE(self): return self.SSE / ((self.n - 1) * (self.k - 1))

  @Nomear.property()
  def estimated_sigma_r(self):
    if self.type == self.Types.ICC2A:
      return np.sqrt((self.MSBS - self.MSE) / self.k)
    else: raise NotImplementedError

  @Nomear.property()
  def estimated_sigma_c(self):
    if self.type == self.Types.ICC2A:
      return np.sqrt((self.MSBM - self.MSE) / self.n)
    else: raise NotImplementedError

  @Nomear.property()
  def estimated_sigma_v(self):
    if self.type == self.Types.ICC2A: return np.sqrt(self.MSE)
    else: raise NotImplementedError

  @Nomear.property()
  def ICC(self):
    if self.type == self.Types.ICC2A:
      return (self.MSBS - self.MSE) / (self.MSBS + (self.k - 1) * self.MSE + (
          self.k / self.n) * (self.MSBM - self.MSE))
    else: raise NotImplementedError

  # endregion: Properties

  # region: Public Methods

  def report(self, verbose=False):
    console.show_info('Data Matrix Info:')
    n, k = self.data.shape
    console.supplement(f'Number of subjects = {n}', level=2)
    console.supplement(f'Number of measurements = {k}', level=2)

    if verbose:
      console.supplement(f'MSBS = {self.MSBS:.3f}')
      console.supplement(f'MSBM = {self.MSBM:.3f}')
      console.supplement(f'MSE = {self.MSE:.3f}')
      if self.type == self.Types.ICC2A:
        console.supplement(
          f'Estimated sigma_r = {self.estimated_sigma_r:.3f}')
        console.supplement(
          f'Estimated sigma_c = {self.estimated_sigma_c:.3f}')
        console.supplement(
          f'Estimated sigma_v = {self.estimated_sigma_v:.3f}')
      else: raise NotImplementedError

    console.supplement(f'{self.type} = {self.ICC:.3f}')

  # endregion: Public Methods



if __name__ == '__main__':
  # Generate data
  n_subjects = n = 20
  n_measurements = k = 3

  # Set model
  mu = 170.
  sigma_r = 10.
  sigma_c = 5.
  sigma_v = 1.
  rho_2A = (sigma_r ** 2) / (sigma_r ** 2 + sigma_c ** 2 + sigma_v ** 2)
  console.show_info(f'Expected ICC(A, 1) = {rho_2A:.3f}')

  # Generate data matrix
  randn = np.random.randn
  data_matrix = mu + sigma_r * randn(n, 1) + sigma_c * randn(1, k)
  data_matrix += sigma_v * randn(n, k)

  data_matrix = np.array([
    [6, 4.1],
    [4, 6],
  ])

  # Calculate ICC(A, 1)
  icc = ICC(data_matrix=data_matrix)
  icc.report(verbose=True)
