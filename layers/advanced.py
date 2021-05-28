from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker, linker
from tframe import hub, context, console
from tframe import initializers

from tframe.layers.layer import LayerWithNeurons, Layer, single_input


class Dense(LayerWithNeurons):

  full_name = 'dense'
  abbreviation = 'dense'

  def __init__(
      self,
      num_neurons,
      activation=None,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      prune_frac=0,
      **kwargs):
    # Call parent's constructor
    LayerWithNeurons.__init__(
      self, activation, weight_initializer, use_bias, bias_initializer,
      prune_frac=prune_frac, **kwargs)

    self.num_neurons = checker.check_positive_integer(num_neurons)
    self.neuron_scale = [num_neurons]

  @property
  def structure_tail(self):
    activation = ''
    if self._activation is not None:
      activation = '->act'
      if isinstance(self._activation_string, str):
        activation = '->' + self._activation_string
    return '({})'.format(self.num_neurons) + activation

  def forward(self, x, **kwargs):
    return self.dense(self.num_neurons, x, activation=self._activation,
                      scope='dense')


class SparseAffine(Layer):
  is_nucleus = True

  full_name = 'sparse_affine'
  abbreviation = 'sparse'

  def __init__(
      self,
      num_neurons,
      heads=1,
      use_bit_max=False,
      logits_initializer='random_normal',
      coef_initializer='random_normal',
      use_bias=True,
      bias_initializer='zeros',
      **kwargs):

    self.num_neurons = checker.check_positive_integer(num_neurons)
    self.heads = checker.check_positive_integer(heads)
    self.use_bit_max = checker.check_type(use_bit_max, bool)
    self._logits_initializer = initializers.get(logits_initializer)
    self._coef_initializer = initializers.get(coef_initializer)
    self._use_bias = checker.check_type(use_bias, bool)
    self._bias_initializer = initializers.get(bias_initializer)

    self.neuron_scale = [self.num_neurons]
    self._kwargs = kwargs

  @property
  def structure_tail(self):
    return '({}->{})'.format(self.heads, self.num_neurons)

  @single_input
  def _link(self, x, **kwargs):
    y, pkg = linker.sparse_affine(
      x, self.num_neurons, self.heads, self.use_bit_max,
      self._logits_initializer, self._coef_initializer, self._use_bias,
      self._bias_initializer, return_package=True)

    # Encourage softmax activation to be saturated
    ds_penalty = self._kwargs.get('desaturate_penalty', 0.0)
    if ds_penalty > 0:
      a = pkg['activation']
      a_bar = tf.subtract(1.0, a)
      context.add_loss_tensor(
        ds_penalty * tf.reduce_mean(tf.minimum(a, a_bar)))
      console.show_status('Desaturate penalty added in {}'.format(
        tf.get_variable_scope().name), '++')

    # Export variables
    if hub.export_sparse_weights:
      scope = '/'.join(tf.get_variable_scope().name.split('/')[1:])
      # context.variables_to_export[scope + '/weights'] = pkg['weights']
      # context.variables_to_export[scope + '/coef'] = pkg['coef']
      context.weights_list.append(pkg['weights'])

    return y
