from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker, linker
from tframe import hub, context
from tframe import initializers
from tframe.layers.layer import LayerWithNeurons
from tframe.operators.apis.distributor import Distributor


class Highway(LayerWithNeurons):

  full_name = 'highway'
  abbreviation = 'hw'

  def __init__(
      self,
      layer_width,
      num_layers,
      activation='relu',
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      t_bias_initializer=-1,
      **kwargs):
    # Call parent's constructor
    LayerWithNeurons.__init__(self, activation, weight_initializer,
                              use_bias, bias_initializer, **kwargs)

    self._layer_width = checker.check_positive_integer(layer_width)
    self._num_layers = checker.check_positive_integer(num_layers)
    assert isinstance(activation, str)
    self._activation_string = activation
    self._t_bias_initializer = initializers.get(t_bias_initializer)

  @property
  def structure_tail(self):
    tail = '{}'.format(self._layer_width)
    if self._activation_string:
      tail = '{}=>{}'.format(tail, self._activation_string)
    return '(({})x{})'.format(tail, self._num_layers)

  def forward(self, x, **kwargs):
    # Alignment check
    assert self.get_dimension(x) == self._layer_width
    # Calculate output iteratively
    for i in range(self._num_layers):
      scope = lambda pre: '{}_{}'.format(pre, i + 1)
      h = self.dense(self._layer_width, x, scope('h'), self._activation)
      t = self.dense_v2(self._layer_width, scope('t'), x, is_gate=True,
                        bias_initializer=self._t_bias_initializer)
      x = h * t + x * (1. - t)
    return x


class LinearHighway(LayerWithNeurons, Distributor):

  full_name = 'linear_highway'
  abbreviation = 'fchw'

  def __init__(
      self,
      output_dim=None,
      spatial_configs=None,
      reverse=False,
      activation='relu',
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      **kwargs):

    assert isinstance(activation, str)
    self.activation_string = activation
    # Call parent's constructor
    LayerWithNeurons.__init__(
      self, activation, weight_initializer, use_bias, bias_initializer,
      **kwargs)

    assert not (output_dim is None and spatial_configs is None)
    self._spatial_groups = []
    if spatial_configs is not None:
      self._spatial_groups = self._get_groups(spatial_configs)
      total_size = self._get_total_size(self._spatial_groups)
      if output_dim is None: output_dim = total_size
      assert output_dim == total_size
    self._output_dim = checker.check_positive_integer(output_dim)
    self._reverse = checker.check_type(reverse, bool)

    self.neuron_scale = [output_dim]

  @property
  def structure_tail(self):
    if len(self._spatial_groups) == 0: tail = '{}'.format(self._output_dim)
    else: tail = self._get_config_string(self._spatial_groups, self._reverse)

    tail = '({})'.format(tail)
    if self._activation is not None:
      tail += '->{}'.format(self.activation_string)
    return tail

  def _get_coupled_gates(self, x, configs, reverse):
    assert isinstance(configs, (list, tuple))
    # T for transform, C for carry
    net_T = self.neurons(x, scope='net_u')
    if len(configs) == 0: T = tf.sigmoid(net_T, name='transform_gate')
    else: T = linker.softmax_over_groups(net_T, configs, 'transform_gate')

    C = tf.subtract(1., T, name='carry_gate')
    if len(configs) > 0 and reverse: T, C = C, T
    if hub.export_gates: self.tensors_to_export['carry_gate'] = C
    return T, C

  def forward(self, x, **kwargs):
    assert linker.get_dimension(x) == self._output_dim
    # Calculate candidates
    H = self.neurons(x, activation=self._activation, scope='y_bar')
    # Calculate transform gate
    T, C = self._get_coupled_gates(x, self._spatial_groups, self._reverse)
    # Calculate y
    y = tf.add(tf.multiply(T, H), tf.multiply(C, x), name='y')

    return y

  @staticmethod
  def extractor(net):
    if not hub.export_gates: return
    from tframe.nets.net import Net
    assert isinstance(net, Net)
    current_dim = -1
    blocks = []
    indices = []
    for i, layer in enumerate(net.layers):
      if not isinstance(layer, LinearHighway): continue
      dim = layer._output_dim
      if dim != current_dim:
        blocks.append([])
        indices.append([])
      blocks[-1].append(layer.tensors_to_export['carry_gate'])
      indices[-1].append(i)
      current_dim = dim
    if len(blocks) == 0: return
    for ids, gates in zip(indices, blocks):
      key = 'carry_gate {}'.format(ids[0])
      if len(ids) > 1: key += '-{}'.format(ids[-1])
      # gate.shape = [bs, dim]
      tensor = tf.stack(gates, axis=1)
      # Register tensor to context
      context.tensors_to_export[key] = tensor









