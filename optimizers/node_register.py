from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.nets.net import Net
from tframe.nets.rnet import RNet
from tframe.layers.layer import Layer


class NodeRegister(object):
  """
  """
  def __init__(self, model):
    # Attributes
    assert isinstance(model, Net)
    self._model = model
    self.blocks = []

    self._census()

  # region : Properties

  @property
  def custom_var_list(self):
    var_list = []
    for block in self.blocks:
      if block.has_customized_gradient:
        var_list += block.container.custom_var_list
    return var_list

  @property
  def default_var_list(self):
    return list(set(self._model.parameters) - set(self.custom_var_list))

  # endregion : Properties

  # region : Public Methods

  def compute_customized_gradient(self, dL_dy):
    """Compute customized gradient
    :param dL_dy: dL/dy \in R^(num_steps(=1), batch_size, *(y.shape))
    :return: (grads_and_vars, new_grad_buffer)
    """
    assert isinstance(dL_dy, tf.Tensor)

    # Compute gradient one by one
    grads_and_vars = []
    new_grad_buffer = []
    for block in [b for b in self.blocks if b.has_customized_gradient]:
      compute_gradients = block.container.compute_gradients
      assert callable(compute_gradients)
      g_n_v, buffer = compute_gradients(dL_dy)
      grads_and_vars += g_n_v
      new_grad_buffer += buffer

    return grads_and_vars, tuple(new_grad_buffer)

  # endregion : Public Methods

  # region : Private Methods

  def _census(self):
    for block in self._model.children:
      if isinstance(block, RNet):
        self.blocks.append(Block(block))
      elif isinstance(block, Net):
        for layer in block.children:
          assert isinstance(layer, Layer)
          if layer.is_nucleus:
            self.blocks.append(Block(layer))
      else:
        raise TypeError('!! Unknown block type {}'.format(type(block)))

  # endregion : Private Methods


class Block(object):
  """"""
  def __init__(self, container):
    assert isinstance(container, (RNet, Layer))
    self.container = container
    self.dynamic_nodes = []
    self.variables = []

    self._init_nodes_n_variables()

  @property
  def has_customized_gradient(self):
    return (isinstance(self.container, RNet) and
            self.container.compute_gradients is not None)

  @property
  def repeater_tensor(self):
    return None

  def _init_nodes_n_variables(self):
    # Get all variables
    self.variables = self.container.parameters



