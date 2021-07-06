import numpy as np
from tframe import tf


def rotate_coord(coord: tf.Tensor, angle):
  """Rotate a batch of coord by different angles

  :param coord: coordinate, shape = [?, 2]
  :param angle: angle to rotate (by angle * 2 * PI), shape = [C]
  :return rotated coord, shape = [?, C, 2]
  """
  # Convert angle to radian (C,)
  radian = 2 * np.pi * angle

  # Generate rotation matrix
  #: (C,), (C,)
  sin, cos = tf.sin(radian), tf.cos(radian)
  cssc = [[cos, -sin], [sin, cos]]
  #: (C, 2, 2)
  M: tf.Tensor = tf.stack([tf.stack(s, -1) for s in cssc], -1)

  # Manually broadcast
  #: (1, C, 2, 2) as a
  M = tf.expand_dims(M, 0)
  #: (?, 1, 2)
  coord = tf.expand_dims(coord, 1)
  #: (?, 1, 2, 1) as b
  coord = tf.expand_dims(coord, -1)

  #: (?, C, 2, 1)
  coord = tf.matmul(M, coord)

  # return coordinate of shape (?, C, 2)
  return tf.squeeze(coord, -1)


if __name__ == '__main__':
  from tframe import console
  console.suppress_logging()

  tf.InteractiveSession()

  np_coord = np.array([[np.sqrt(2), np.sqrt(2)], [1, 0]])
  np_angles = [1/8, 1/4]

  coord = tf.constant(np_coord, dtype=tf.float32)
  angles = tf.constant(np_angles, dtype=tf.float32)
  coord = rotate_coord(coord, angles)

  for i, c in enumerate(np_coord):
    print(f'Origin coord = {c}')
    for j, a in enumerate(np_angles):
      print(f'rotated by {a * 360} degree')
      console.eval_show(coord[i, j])






