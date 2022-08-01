from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from tframe import tf
from tframe.nets.net import Net
from tframe.layers.layer import Layer



class Supotco(Net):
  """User obligation:
  (1) setting gates
  (2) ...

  Minimal demo
  ------------
    from tframe import mu
    from tframe import Predictor
    from tframe.nets.supotco import Supotco

    import numpy as np



    # Build model
    model = Predictor(mark='Supotco')

    model.add(mu.Input([32]))
    model.add(mu.Dense(16, activation='relu'))

    su: Supotco = model.add(Supotco())
    li = su.init_a_limb('output-1')
    li.add(mu.Dense(10, activation='softmax'))

    li = su.init_a_limb('output-2')
    li.add(mu.Dense(10, activation='softmax'))

    su.set_gates([1, 0])

    model.rehearse(export_graph=True)


    # Predict
    from tframe import DataSet
    data = DataSet(np.zeros([1, 32]))
    output = model.predict(data)
  """

  def __init__(self, name='Supotco'):
    # Call parent's initializer
    super(Supotco, self).__init__(name)

    self.gates: list = []

  # region: Properties

  @property
  def structure_detail(self):
    rows, total_params, dense_total = [], 0, 0
    for w, child in zip(self.gates, self.children):
      assert isinstance(child, Net)
      # Get details
      _rows, num, dense_num = child.structure_detail
      # Add indentation to each layer
      for i, row in enumerate(_rows):
        prefix = ' ' * 3 if i > 0 else f'{"=" if w > 0 else "x"}> '
        # Add indentation according to level
        prefix = ' ' * 3 * (self._level - 1) + prefix
        # Prefix row
        _rows[i][0] = prefix + row[0]
      # Accumulate stuff
      rows += _rows
      total_params += num
      dense_total += dense_num

    # Return
    return rows, total_params, dense_total

  def structure_string(self, detail=True, scale=True):
    return '{}({})'.format(self.name, ', '.join(
      [f'{"[x]" if w == 0 else ""}' + child.structure_string(detail, scale)
       for child, w in zip(self.children, self.gates)]))

  @Net.property()
  def limbs(self): return OrderedDict()

  # endregion: Properties

  # region: Public Methods

  def init_a_limb(self, limb_key):
    # Make sure `limb_key` has not been registered yet
    if limb_key in self.limbs:
      raise KeyError(f'Key `{limb_key}` already exists')

    # Create a new limb as a net and return
    return self._get_limb(limb_key)

  def add_to_limb(self, limb_key, func):
    # Sanity check then get limb
    assert isinstance(func, (Layer, Net))
    return self._get_limb(limb_key).add(func)

  def set_gates(self, gates):
    assert len(gates) == len(self.children)
    self.gates = gates

  # endregion: Public Methods

  # region: Private Methods

  def _get_limb(self, key) -> Net:
    if key not in self.limbs: self.limbs[key] = self.add(name=key)
    return self.limbs[key]

  # endregion: Private Methods

  # region: APIs

  def _link(self, x: tf.Tensor, **kwargs):
    # Sanity check
    assert len(self.gates) == len(self.children)

    # Forward each child
    tensors = [child(x) for child in self.children]

    # Merge
    if sum(self.gates) != 1:
      raise AssertionError('!! Currently only one output is supported')

    return [tensor for w, tensor in zip(self.gates, tensors) if w][0]

  # endregion: APIs
