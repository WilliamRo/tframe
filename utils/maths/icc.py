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
  def sr2(self):
    if self.type == self.Types.ICC1:
      return (self.MSBS - self.MSWS) / self.k
    return (self.MSBS - self.MSE) / self.k

  @Nomear.property()
  def sr(self): return np.sqrt(self.sr2)

  @Nomear.property()
  def sc2(self):
    if self.type == self.Types.ICC2A:
      return (self.MSBM - self.MSE) / self.n
    else: raise NotImplementedError

  @Nomear.property()
  def sc(self): return np.sqrt(self.sc2)

  @Nomear.property()
  def sv2(self):
    if self.type == self.Types.ICC1: return self.MSWS
    return self.MSE

  @Nomear.property()
  def sv(self): return np.sqrt(self.sv2)

  @Nomear.property()
  def ICC(self):
    if self.type in (self.Types.ICC1, self.Types.ICC2C, self.Types.ICC3C):
      return self.sr2 / (self.sr2 + self.sv2)
    if self.type in (self.Types.ICC2A, self.Types.ICC3A):
      return self.sr2 / (self.sr2 + self.sc2 + self.sv2)
    else: raise NotImplementedError

  # endregion: Properties

  # region: Public Methods

  def report(self, verbose=False, d=3):
    console.show_info('Data Matrix Info:')
    n, k = self.data.shape
    console.supplement(f'Number of subjects = {n}', level=2)
    console.supplement(f'Number of measurements = {k}', level=2)

    if verbose:
      console.supplement(f'MSBS = {self.MSBS:.{d}f}')
      console.supplement(f'MSBM = {self.MSBM:.{d}f}')
      console.supplement(f'MSWS = {self.MSWS:.{d}f}')
      console.supplement(f'MSWM = {self.MSWM:.{d}f}')
      console.supplement(f'MSE = {self.MSE:.{d}f}')
      if self.type in (self.Types.ICC1, self.Types.ICC2C, self.Types.ICC3C):
        console.supplement(f'Estimated sigma_r^2 = {self.sr2:.{d}f}')
        console.supplement(f'Estimated sigma_v^2 = {self.sv2:.{d}f}')
      elif self.type in (self.Types.ICC2A, self.Types.ICC3A):
        console.supplement(f'Estimated sigma_r^2 = {self.sr2:.{d}f}')
        console.supplement(f'Estimated sigma_c^2 = {self.sc2:.{d}f}')
        console.supplement(f'Estimated sigma_v^2 = {self.sv2:.{d}f}')
      else: raise NotImplementedError

    console.supplement(f'{self.type} = {self.ICC:.{d}f}')

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

  data_matrix_1 = np.array([[6, 6.1], [3, 2.9], [4, 4.2], [5, 4.8]])
  data_matrix_2 = np.array([[6, 6.1], [6, 5.9], [6, 6.2], [6, 5.8]])
  data_matrix_3 = np.array([[6.01, 6.], [6, 6.01], [6, 6.01], [6, 6.01]])

  # Calculate ICC
  icc = ICC(data_matrix_3, ICC.Types.ICC2A)
  icc.report(verbose=True, d=5)
