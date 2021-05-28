from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf


class Function(object):
  """A core concept in tframe"""
  master = None
  superior = None

  parameters = None
  linked = False
  output_tensor = None

  output_id = None  # this attribute is for shortcut

  @property
  def output_id_str(self):
    # if not self.output_id: return ''
    if not self.output_id: self.set_output_id()
    return '[{}]'.format(self.output_id)

  @property
  def output_shape_str(self):
    tensor = self.output_tensor
    if tensor is None: return ''
    assert isinstance(tensor, tf.Tensor)
    shape_list = tensor.shape.as_list()
    return 'x'.join(['{}'.format(d) for d in shape_list[1:]])

  def group_name(self):
    raise NotImplementedError('Property "group_name" has not implemented yet')

  def __call__(self, *inputs, **kwargs):
    """When a Function is called, it will be linked into a model and the
       corresponding parameters are registered

    :return: the output tf tensor
    """
    # Get the link method
    link = lambda: self._link(*inputs, **kwargs)
    # Handle the situation when self can be both feed-forward and recurrent
    if self.master is not None:
      assert issubclass(self.master, Function)
      link = lambda: self.master._link(self, *inputs, **kwargs)

    # Call _link to get the output tensor and register parameters
    def get_output_and_register():
      output = link()
      self.parameters = tf.trainable_variables(tf.get_variable_scope().name)
      return output

    if self.group_name is not None:
      with tf.variable_scope(self.group_name, reuse=tf.AUTO_REUSE):
        output = get_output_and_register()
    else:
      output = get_output_and_register()

    self.linked = True
    if isinstance(output, (list, tuple)):
      self.output_tensor = output[0]
    else:
      assert isinstance(output, tf.Tensor)
      self.output_tensor = output
    return output


  def _link(self, *inputs, **kwargs):
    raise NotImplementedError('_link method not implemented')

  def set_output_id(self):
    from tframe import context
    if self.output_id is None:
      self.output_id = context.get_next_output_id()
