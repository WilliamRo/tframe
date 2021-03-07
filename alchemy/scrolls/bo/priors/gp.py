import sklearn

from .prior_base import PriorBase


class GaussianProcess(PriorBase):

  def __init__(self, **kwargs):
    super(GaussianProcess, self).__init__(**kwargs)

    # Get kernel
    self.kernel = None
    # Create model
    self.model = sklearn.gaussian_process.GaussianProcessRegressor(
      self.kernel, **kwargs)


  def feed_observations(self, x_list, y_list):
    pass


class Kernel(object):
  pass



