from tframe.core import Group
from tframe.core import TensorSlot, OperationSlot
from tframe.layers.common import Input
from tframe.models.model import Model
from tframe.nets.net import Net
from tframe import context
from tframe import DataSet
from tframe import hub as th
from tframe import pedia
from tframe import tf

import numpy as np



class VAE(Model):
  """Variational Autoencoders"""

  def __init__(self, mark, input_shape, latent_dim):
    # Call parent's constructor
    super().__init__(mark)

    # Define encoder and decoder
    self.Encoder = Net(pedia.Encoder)
    self.Decoder = Net(pedia.Decoder)
    # Alias
    self.E = self.Encoder
    self.D = self.Decoder

    # Add input layers
    self.E.add(Input(sample_shape=input_shape, name='input'))
    self.D.add(Input(sample_shape=[latent_dim], name=pedia.D_input))

  # region : Properties

  @property
  def description(self):
    return [f'Encoder: {self.E.structure_string()}',
            f'Decoder: {self.D.structure_string()}']

  # endregion : Properties

  # region: Public Methods

  def generate(self, z=None, sample_num=1):
    # Check model and session
    if not self.D.linked: raise AssertionError('!! Model not built yet.')
    if not self.launched: self.launch_model(overwrite=False)

    # Get sample number
    sample_num = (th.sample_num if th.sample_num > 0 else max(1, sample_num))

    # Check input z
    z = self._random_z(sample_num) if z is None else z
    z_shape = list(z.shape[1:])
    d_input_shape = self.D.input_tensor.get_shape().as_list()[1:]
    if z_shape != d_input_shape:
      raise ValueError("!! Shape of input z {} doesn't match the shape of "
                       "decoder's input {}".format(z_shape, d_input_shape))

    # Generate samples
    feed_dict = {self.D.input_tensor: z}
    feed_dict.update(self.agent.get_status_feed_dict(is_training=False))
    samples = self.outputs.run(feed_dict)
    return samples

  # endregion: Public Methods

  # region: Private Methods

  def _build(self, optimizer=None, loss=pedia.cross_entropy, **kwargs):
    pass

  def _define_losses(self, loss, alpha):
    pass

  def _random_z(self, sample_num):
    z_dim = self.G.input_tensor.get_shape().as_list()[1]
    z = np.random.standard_normal(size=[sample_num, z_dim])
    return z

  # endregion: Private Methods

  # region: Overwriting

  def update_model(self, data_batch, **kwargs):
    assert isinstance(data_batch, DataSet)

    pass

  def handle_structure_detail(self):
    E_rows, E_total_params, E_dense_total = self.E.structure_detail
    D_rows, D_total_params, D_dense_total = self.D.structure_detail

    # Take some notes
    params_str = 'Encoder total params: {}'.format(E_total_params)
    self.agent.take_notes(params_str)
    params_str = 'Decoder total params: {}'.format(D_total_params)
    self.agent.take_notes(params_str)

    if th.show_structure_detail:
      print('.. Encoder structure detail:\n{}'.format(E_rows))
      print('.. Decoder structure detail:\n{}'.format(D_rows))

    if th.export_structure_detail:
      self.agent.take_notes('Structure detail of Encoder:', False)
      self.agent.take_notes(E_rows, False)
      self.agent.take_notes('Structure detail of Decoder:', False)
      self.agent.take_notes(D_rows, False)

  # endregion: Overwriting

