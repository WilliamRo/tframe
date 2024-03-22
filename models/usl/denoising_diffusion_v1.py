from tframe.layers.common import Input
from tframe.layers.hyper.hyper_base import HyperBase
from tframe.models import Predictor
from tframe import DataSet
from tframe import hub as th
from tframe import tf
from tframe import pedia

import numpy as np



class GaussianDiffusion(Predictor):
  """Gaussian Diffusion Model
  Ref: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
  """

  def __init__(
      self,
      mark,
      x_shape,
      time_steps=1000,
      sampling_time_steps=1000,
      time_dim=128,
      beta_schedule='cosine',
      schedule_fn_kwargs=dict(),
      **kwargs
  ):
    # Call parent's constructor
    super().__init__(mark)

    # Attributes
    self.time_steps = time_steps
    self.sampling_time_steps = sampling_time_steps
    self.configs = kwargs

    # Placeholders
    self.add(Input(sample_shape=x_shape))
    self.time_table = self.get_sinusoidal_pos_emb(
      np.arange(self.time_steps), time_dim)
    self.time_emb = tf.placeholder(
      dtype=th.dtype, shape=[None, time_dim], name='time_emb', )
    tf.add_to_collection(pedia.default_feed_dict, self.time_emb)

    # Calculate alpha related stuff
    self.betas = self.generate_schedule(beta_schedule, **schedule_fn_kwargs)

  # region : Properties

  @Predictor.property()
  def alphas(self): return 1. - self.betas

  @Predictor.property()
  def bar_alphas(self): return np.cumprod(self.alphas, axis=0)

  @Predictor.property()
  def sqrt_bar_alphas(self): return np.sqrt(self.bar_alphas)

  @Predictor.property()
  def sqrt_one_minus_bar_alphas(self): return np.sqrt(1. - self.bar_alphas)

  @Predictor.property()
  def sqrt_alphas(self): return np.sqrt(self.alphas)

  # endregion : Properties

  # region: Diffusion Library

  # region: Batch preprocessor

  def batch_preprocessor(self, data_batch, is_training):
    """See DDPM paper -> Algorithm 1"""
    from tframe import DataSet
    assert is_training and isinstance(data_batch, DataSet)

    # Generate t (shape = (N, 1))
    x_0: np.ndarray = data_batch.features
    # TODO: t starts from 0 or 1?
    t = np.random.randint(0, self.time_steps, size=data_batch.size)
    epsilon = np.random.randn(*x_0.shape)
    indices = list(t)
    t_shape = [-1] + (len(x_0.shape) - 1) * [1]

    sqrt_bar_alpha_t = self.sqrt_bar_alphas[indices]
    sqrt_bar_alpha_t = sqrt_bar_alpha_t.reshape(t_shape)
    sqrt_one_minus_bar_alpha_t = self.sqrt_one_minus_bar_alphas[indices]
    sqrt_one_minus_bar_alpha_t = sqrt_one_minus_bar_alpha_t.reshape(t_shape)

    x_t = (sqrt_bar_alpha_t * x_0 + sqrt_one_minus_bar_alpha_t * epsilon)

    # Set x_t, epsilon, t to data_batch
    data_batch.features = x_t
    data_batch.targets = epsilon
    data_batch.data_dict['time_emb'] = self.time_table[indices].astype(
      np.float)

    return data_batch

  # endregion: Batch preprocessor

  # region: Schedules

  def generate_schedule(self, beta_schedule, **kwargs):
    betas = {
      'linear': self.linear_beta_schedule,
      'cosine': self.cosine_beta_schedule,
      'cos': self.cosine_beta_schedule,
      'sigmoid': self.sigmoid_beta_schedule,
    }[beta_schedule](self.time_steps, **kwargs)
    return betas

  @staticmethod
  def linear_beta_schedule(time_steps):
    """linear schedule, proposed in original DDPM paper"""
    scale = 1000 / time_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, time_steps, dtype=np.float)

  @staticmethod
  def cosine_beta_schedule(time_steps, s=0.008):
    """cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = time_steps + 1
    t = np.linspace(0, time_steps, steps, dtype=np.float) / time_steps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

  @staticmethod
  def sigmoid(x): return 1 / (1 + np.exp(-x))

  @classmethod
  def sigmoid_beta_schedule(cls, time_steps, start=-3, end=3, tau=1):
    """sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = time_steps + 1
    t = np.linspace(0, time_steps, steps, dtype=np.float) / time_steps
    v_start, v_end = [cls.sigmoid(x / tau) for x in (start, end)]
    alphas_cumprod = -cls.sigmoid((t * (end - start) + start) / tau) + v_end
    alphas_cumprod = alphas_cumprod / (v_end - v_start) / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

  # endregion: Schedules

  # endregion: Diffusion Library

  # region: Sampling Methods

  def _sample_with_clip(self, x_t, t, pred_noise):
    # Calculate predicted image
    coef_shape = [-1] + [1] * (len(pred_noise.shape) - 1)
    alpha_t = self.alphas[t].reshape(coef_shape)
    bar_alpha_t = self.bar_alphas[t].reshape(coef_shape)
    beta_t = self.betas[t].reshape(coef_shape)

    # Predict x_0 using pred_noise
    x_0_pred = np.sqrt(1. / bar_alpha_t) * x_t - np.sqrt(
      1. / bar_alpha_t - 1) * pred_noise
    x_0_pred = np.clip(x_0_pred, -1., 1.)

    z = np.random.randn(x_t.shape[0], *th.input_shape)
    if t > 0:
      sqrt_bar_alpha_t_prev = self.sqrt_bar_alphas[t - 1].reshape(coef_shape)
      bar_alpha_t_prev = self.bar_alphas[t - 1].reshape(coef_shape)
      sqrt_alpha_t = self.sqrt_alphas[t].reshape(coef_shape)
      mean = (beta_t * sqrt_bar_alpha_t_prev) / ( 1. - bar_alpha_t) * x_0_pred
      mean += ((1. - bar_alpha_t_prev) * sqrt_alpha_t) / (1. - bar_alpha_t) * x_t
      std = np.sqrt(beta_t * (1. - bar_alpha_t_prev) / (1. - bar_alpha_t))
    else:
      mean = (beta_t / (1. - bar_alpha_t)) * x_0_pred
      std = 0.

    return mean + std * z

  def _sample_without_clip(self, x_t, t, pred_noise):
    # Calculate predicted image
    coef_shape = [-1] + [1] * (len(pred_noise.shape) - 1)
    alpha_t = self.alphas[t].reshape(coef_shape)
    bar_alpha_t = self.bar_alphas[t].reshape(coef_shape)
    beta_t = self.betas[t].reshape(coef_shape)

    sqrt_one_minus_bar_alpha_t = self.sqrt_one_minus_bar_alphas[t].reshape(
      coef_shape)
    sqrt_alpha_t = self.sqrt_alphas[t].reshape(coef_shape)

    gamma = (1. - alpha_t) / sqrt_one_minus_bar_alpha_t
    mu = (x_t - gamma * pred_noise) / sqrt_alpha_t

    if t == 0: return mu

    z = np.random.randn(x_t.shape[0], *th.input_shape)
    # Calculate sigma
    bar_alpha_t_prev = self.bar_alphas[t - 1].reshape(coef_shape)
    sigma = np.sqrt(beta_t * (1. - bar_alpha_t_prev) / (1. - bar_alpha_t))

    mu += z * sigma
    return mu

  def generate(self, sample_num=1, x_T=None, return_all_images=False,
               clip=True):
    """See DDPM paper -> Algorithm 2
    Ref: https://github.com/bot66/MNISTDiffusion/blob/main/train_mnist.py
    """
    x_t = np.random.randn(sample_num, *th.input_shape) if x_T is None else x_T
    images = [x_t]
    for t in reversed(range(self.time_steps)):
      # Calculate predicted epsilon_theta
      time_emb = self.time_table[[t] * sample_num]
      pred_noise = self.predict(DataSet(data_dict={
        'features': x_t, 'time_emb': time_emb}))

      if clip: x_t = self._sample_with_clip(x_t, t, pred_noise)
      else: x_t = self._sample_without_clip(x_t, t, pred_noise)

      images.append(x_t)

    # TODO: ...? be very careful
    images = [(x + 1.) / 2. for x in images]

    if return_all_images: return images
    return x_t

  # endregion: Sampling Methods

  # region: Time Embedding

  class TimeEmbedding(HyperBase):

    abbreviation = 'time_emb'
    full_name = 'time_emb'

    def __init__(self, time_emb, hidden_dim=None, activation='relu'):
      super().__init__(activation=activation)
      self.time_emb: tf.Tensor = time_emb
      if hidden_dim is None: hidden_dim = time_emb.shape.as_list()[-1]
      self.hidden_dim = hidden_dim

    @property
    def structure_tail(self):
      return f'({self.hidden_dim}, {self._activation_string})'

    def _link(self, x: tf.Tensor, **kwargs):
      C = x.shape.as_list()[-1]
      te = self.dense(self.hidden_dim, self.time_emb, 'time_hid',
                      activation=self._activation)
      te = self.dense(C, te, 'time_emb_to_add')
      while len(te.shape) < len(x.shape):
        te = tf.expand_dims(te, axis=1)

      return te + x

  def get_time_emb_layer(self):
    return self.TimeEmbedding(self.time_emb)

  # endregion: Time Embedding

  # region: Public Methods

  def generate_demo(self, sample_num=16, clip=True):
    from pictor import Pictor
    from tframe.utils import imtool

    def plot(fig, x): imtool.gan_grid_plot(x, fig=fig)

    p = Pictor('DDPM Demo')
    p.add_plotter(plot)
    p.objects = self.generate(sample_num, return_all_images=True, clip=clip)
    p.show()

  # endregion: Public Methods

  # region: MISC

  @staticmethod
  def get_sinusoidal_pos_emb(t: np.ndarray, dim, theta=10000):
    """Modified from lucidrains' SinusoidalPosEmb"""
    assert len(t.shape) == 1
    half_dim = dim // 2
    emb = np.log(theta) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = np.concatenate((np.sin(emb), np.cos(emb)), axis=-1)
    return emb

  # endregion: MISC



if __name__ == '__main__':
  from tframe import console

  def test_pos_emb():
    import matplotlib.pyplot as plt
    t = np.arange(100)
    plt.imshow(GaussianDiffusion.get_sinusoidal_pos_emb(t, 100))
    plt.show()

  test_pos_emb()



