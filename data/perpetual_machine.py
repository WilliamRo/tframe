from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import checker
from tframe import hub

from tframe.data.base_classes import TFRData
from tframe.data.dataset import DataSet
from tframe.data.sequences.seq_set import SequenceSet


class PerpetualMachine(TFRData):

  def __init__(self, name, engine, **kwargs):
    """Construct a `perpetual machine`
    :param name: name
    :param engine: a function accepts `size` as input. engine takes care of
                   properties like `n_to_one`.
    """
    # Call parent't constructor
    super().__init__(name)
    # Check input
    assert callable(engine)
    self.engine = engine
    self.generate_sequence = None
    # Force to examine engine
    self._examine_engine()
    # Set property
    self.properties = kwargs

  # region : Properties

  @property
  def is_regular_array(self): return False

  # endregion : Properties

  # region : APIs

  def get_round_length(self, *args, **kwargs):
    return None

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    checker.check_positive_integer(batch_size)
    while True:
      # gen_batches for sequences is not supported yet
      assert not self.generate_sequence
      yield self.engine(batch_size)

  def gen_rnn_batches(self, batch_size=1, num_steps=-1, shuffle=False,
                      is_training=False):
    checker.check_positive_integer(batch_size)
    while True:
      data_set = self.engine(batch_size)
      assert isinstance(data_set, DataSet)
      for batch in data_set.gen_rnn_batches(batch_size, num_steps):
        yield batch

  # endregion : APIs

  # region : Public Methods

  # endregion : Public Methods

  # region : Private Methods

  def _examine_engine(self):
    N = 2
    data_batch = self.engine(N)
    assert isinstance(data_batch, DataSet)
    self.generate_sequence = isinstance(data_batch, SequenceSet)
    if data_batch.size != N:
      raise AssertionError('!! Meant to generate a data set of size {} but '
                           'get a size {}'.format(N, data_batch.size))

  # endregion : Private Methods
