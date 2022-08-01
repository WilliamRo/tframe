from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from tframe import tf
from tframe.nets.net import Net
from tframe.layers.layer import Layer
from tframe.layers.common import Input



class Octopus(Net):
  """User obligation:
  (1) making sure the output of each head has the same shape
  (2) setting gates

  Minimal demo
  ------------
    from tframe import mu
    from tframe import Predictor
    from tframe.nets.octopus import Octopus

    import numpy as np


    # Build model
    model = Predictor(mark='octopus-5')

    oc: Octopus = model.add(Octopus())

    li = oc.init_a_limb('input-1', [32])
    li.add(mu.Dense(8))

    li = oc.init_a_limb('input-2', [16])
    li.add(mu.Dense(8))

    oc.set_gates([1, 0])

    model.add(mu.Activation('relu'))
    model.add(mu.Dense(10, activation='softmax'))
    model.rehearse(export_graph=True)


    # Predict (feed opened gate only)
    from tframe import DataSet
    data = DataSet(data_dict={'input-1': np.zeros([1, 32])})
    output = model.predict(data)
  """

  def __init__(self, name='octopus'):
    # Call parent's initializer
    super(Octopus, self).__init__(name)

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

      # Show output id in structure detail if this branch is enabled
      if w: rows[-1][0] += f':={child.output_id_str}'

    # Return
    output_ids = [child.output_id_str for w, child in zip(
      self.gates, self.children) if w]
    rows.append([f'Sum({",".join(output_ids)})', rows[-1][1], ''])
    return rows, total_params, dense_total

  def structure_string(self, detail=True, scale=True):
    return '{}({})'.format(self.name, ', '.join(
      [f'{"[x]" if w == 0 else ""}' + child.structure_string(detail, scale)
       for child, w in zip(self.children, self.gates)]))

  @Net.property()
  def limbs(self): return OrderedDict()

  # endregion: Properties

  # region: Public Methods

  def init_a_limb(self, limb_key, input_shape, input_key=None):
    # Make sure `limb_key` has not been registered yet
    if limb_key in self.limbs:
      raise KeyError(f'Key `{limb_key}` already exists')

    # Set default input_key if not provided
    if input_key is None: input_key = limb_key

    # Create a new limb as a net
    new_limb = self._get_limb(limb_key)

    # Put an input layer to the new limb. The added input layer will be
    #  automatically added to `default_feed_dict` collection
    new_limb.add(Input(sample_shape=input_shape, name=input_key))
    return new_limb

  def add_to_limb(self, limb_key, func):
    # Sanity check then get limb
    assert isinstance(func, (Layer, Net))
    limb = self._get_limb(limb_key)
    if not isinstance(func, Input) and not isinstance(limb.children[0], Input):
      raise AssertionError(f'!! `{limb_key}` has not been initialized yet.')

    return limb.add(func)

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

  def _link(self, *inputs, **kwargs):
    # Sanity check
    assert len(self.gates) == len(self.children)

    # Set dummy input for closed gate
    for w, child in zip(self.gates, self.children):
      if w == 0: child.input_.dummy = True

    # Forward each child
    tensors = [child() for child in self.children]

    # Merge
    if sum(self.gates) == 0: raise AssertionError('!! All inputs are disabled.')
    return tf.add_n([tensor for w, tensor in zip(self.gates, tensors) if w])

  # endregion: APIs
