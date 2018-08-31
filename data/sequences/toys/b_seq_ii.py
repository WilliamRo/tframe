from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.base_classes import DataAgent
from tframe.utils.misc import convert_to_one_hot


class BinarySequenceII(DataAgent):
  """XOR
     TODO: rnn & lstm = 0.5; AMU = 0.55
  """
  DATA_NAME = 'BinarySequenceII'

  @classmethod
  def load_random(cls, train_size=80000, test_size=20000):
    # Generate train set
    x, y = cls._gen_data(train_size)
    train_set = SequenceSet([x], [y], name='TrainSet')

    # Generate test set
    x, y = cls._gen_data(test_size)
    test_set = SequenceSet([x], [y], name='TestSet')

    return train_set, test_set


  @classmethod
  def _gen_data(cls, size):
    assert isinstance(size, int) and size > 0

    x = np.random.choice(2, size=(size,))
    y = []
    for i in range(size):
      if x[i-3] == x[i-8]: y.append(0)
      else: y.append(1)

    return convert_to_one_hot(x, 2), convert_to_one_hot(np.array(y), 2)


if __name__ == '__main__':
  train_set, test_set = BinarySequenceII.load_random()
  a = 1
