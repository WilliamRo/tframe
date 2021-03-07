import sklearn


class PriorBase(object):

  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def feed_observations(self, x_list, y_list):
    raise NotImplementedError