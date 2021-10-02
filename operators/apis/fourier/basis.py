from tframe.utils.maths.misc import rotate_coord as rotate
from tframe import hub as th
from tframe import tf
from tframe import console

from typing import Optional

import numpy as np


_buffers = {}

def get_fourier_basis(
    L: int, center: tf.Tensor, uv: tf.Tensor, r: tf.Tensor, theta: tf.Tensor,
    radius: Optional[tf.Tensor] = None, fmt='r'):
  """Get Fourier basis.

  :param L: odd integer
  :param center: 2-D rank-1 tensor, shape (?, 2)
  :param uv: unit vector, |uv| == aperture, shape = [?, 2]
  :param r: scalar, \in (0, 1), shape = [C]
  :param theta: angle, scalar, \in [0, 360), shape = [C]
  :param radius: pupil radius, \in [0, 1]
  :param fmt: \in {'r', 'i', 'c'}
  :return: tensor of shape [?, L, L, C]
  """
  # Check input parameters
  assert fmt in ('r', 'i', 'c')
  assert isinstance(L, int) and L % 2 == 1

  # Get IJ, shape = [L, L, 2]
  R = (L - 1) // 2
  key = f'IJ(L={L})'
  if key in _buffers: IJ = _buffers[key]
  else:
    #: [L, L, 2]
    IJ = tf.stack([tf.constant(M, dtype=th.dtype) for M in np.meshgrid(
      range(-R, R + 1), range(-R, R + 1), indexing='ij')], axis=-1)
    # Reshape IJ for future computing
    IJ_shape = [1, L, L, 1, 2]
    IJ = tf.reshape(IJ, shape=IJ_shape)
    _buffers[key] = IJ

  # Calculate coord [?, C, 2] using r [?, C] and theta [?, C]
  C = r.shape.as_list()[-1]
  #: [?, 2] -> [?, 1, 2]
  center = tf.expand_dims(center, 1)
  #: [C] -> [1, C, 1]
  r = tf.reshape(r, shape=[1, C, 1])
  # uv.shape = [?, 2], theta.shape = [C]
  # rotate(uv, theta).shape = [?, C, 2]
  #: [?, C, 2]
  coord = center + r * rotate(uv, theta)

  # Calculate basis [?, L, L, C]
  #   using IJ.shape = [L, L, 2], coord.shape = [?, C, 2]
  #                 [1, L, L, 1, 2]             [?, 1, 1, C, 2]
  coord = tf.reshape(coord, shape=[-1, 1, 1, C, 2])
  #: [?, L, L, C]
  real: tf.Tensor = tf.cos(-2*np.pi*tf.reduce_sum(IJ * coord, axis=-1))

  # Calculate imag part if required
  imag: Optional[tf.Tensor] = None
  if fmt in ('c', 'i'):
    imag: tf.Tensor = tf.sin(-2 * np.pi * tf.reduce_sum(IJ * coord, axis=-1))

  # Consider rotundity, radius.shape = [C]
  if radius is not None:
    assert radius.shape.as_list()[-1] == C

    # Get distance matrix D [1, L, L, 1]
    key = f'D(L={L})'
    if key in _buffers: D = _buffers[key]
    else:
      # IJ.shape = [1, L, L, 1, 2]
      D = tf.sqrt(tf.reduce_sum(tf.square(IJ), axis=-1))
      _buffers[key] = D

    # Calculate mask
    #: [C] -> [1, 1, 1, C]
    radius = tf.reshape(radius, shape=[1, 1, 1, C])

    # mask should be differentiable (see 06_softmask.py)
    alpha = 0.1
    beta = np.log(1 - alpha) - np.log(alpha)
    # D [1, L, L, 1], radius [1, 1, 1, C]
    mask = tf.sigmoid(beta * (radius * R - D))

    # Hard mask
    # mask = tf.less_equal(D, R * radius)
    # mask = tf.cast(mask, th.dtype)

    real = tf.multiply(real, mask)
    if imag is not None: imag = tf.multiply(imag, mask)

  # Return specified format
  if fmt == 'r': return real
  if fmt == 'i': return imag
  return (real, imag)


if __name__ == '__main__':
  from lambo import DaVinci
  from tframe import console
  console.suppress_logging()
  tf.InteractiveSession()

  L = 17
  center = tf.constant([[0.1778, -0.2453], [-0.2453, 0.1778]])
  uv = tf.constant([[0.0392, -0.0541], [-0.0541, 0.0392]])
  N = center.shape.as_list()[0]

  da = DaVinci('TFrame fourier', init_as_image_viewer=True)
  sess: tf.Session = tf.get_default_session()

  C = 10
  np_angle = list(np.linspace(0, 1, C))
  r = tf.constant([0.3] * C)
  theta = tf.constant(np_angle)

  # radius = tf.constant([1.0] * C)
  radius = tf.constant(list(np.linspace(1.0, 0.4, C)))

  tensor = get_fourier_basis(L, center, uv, r, theta, radius, fmt='i')
  #: [?, L, L, C]
  basis: np.ndarray = sess.run(tensor)
  assert basis.shape == (2, L, L, C)

  for i in range(N):
    for j in range(C):
      da.add_image(basis[i, :, :, j], title=f'sample[{i}], channel[{j}]')

  da.show()



