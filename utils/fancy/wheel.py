import numpy as np


class Wheel(object):

  def __init__(self, weights):
    assert isinstance(weights, (list, tuple))
    assert all(np.array(weights) >= 0)
    self.weights = weights

  def spin(self):
    number = np.random.rand() * sum(self.weights)
    for index in range(len(self.weights)):
      number -= self.weights[index]
      if number <= 0: return index
    # This is not gonna happen
    assert False

  @staticmethod
  def i_feel_lucky(weights): return Wheel(weights).spin()

i_feel_luck = Wheel.i_feel_lucky


if __name__ == '__main__':
  print(np.random.rand())
