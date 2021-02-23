from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.merge import Merge
from tframe.nets.net import Net


class ForkMerge(Net):
  """A sub-network has the following topology:

       fork           merge
         _______________
       /                \
   -> x ---------------- y ->
      \                /
      ...           ...
       \____________/

  """

  def __init__(self, merge_layer, name='forkmerge', **kwargs):
    # Call parent's initializer
    super(ForkMerge, self).__init__(name, **kwargs)
    # Specialized fields
    if isinstance(merge_layer, Layer): self.merge_layer = merge_layer
    else: self.merge_layer = Merge(merge_layer, **kwargs)
    self._branch_dict = {}
    self._stop_gradient_keys = kwargs.get('stop_gradient_at', [])
    self._kwargs = kwargs


  @property
  def structure_detail(self):
    rows, total_params, dense_total = [], 0, 0
    for child in self.children:
      assert isinstance(child, Net)
      # Set output ID for each child
      child.children[-1].set_output_id()
      # Get details
      _rows, num, dense_num = child.structure_detail
      # Add indentation to each layer
      for i, row in enumerate(_rows):
        prefix = ' ' * 3 if i > 0 else '=> '
        _rows[i][0] = prefix + row[0]
      # Accumulate stuff
      rows += _rows
      total_params += num
      dense_total += dense_num

    # Consider merge layer
    row, num, dense_num = self._get_layer_detail(self.merge_layer)
    row[0] += '({})'.format(','.join(
      [child.children[-1].output_id_str for child in self.children]))
    rows.append(row)
    total_params += num
    dense_total += dense_num
    # Return
    return rows, total_params, dense_total


  def structure_string(self, detail=True, scale=True):
    return '{}({})'.format(self.name, ', '.join(
      [child.structure_string(detail, scale) for child in self.children] +
      [self.merge_layer.get_layer_string(False, False)]))


  def add_to_branch(self, branch_key, layer):
    # Get the Net to add layer
    if branch_key not in self._branch_dict:
      self._branch_dict[branch_key] = self.add(name=str(branch_key))
    branch = self._branch_dict[branch_key]
    # Add layer to branch
    assert isinstance(layer, Layer) and isinstance(branch, Net)
    return branch.add(layer)


  def _link(self, x:tf.Tensor, **kwargs):
    x_stop = tf.stop_gradient(x)
    output_list = [net(x_stop) if key in self._stop_gradient_keys else net(x)
                   for key, net in self._branch_dict.items()]
    return self.merge_layer(output_list)


