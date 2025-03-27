from tframe import tf
from tframe.layers.layer import Layer, single_input

import numpy as np



class Partition(Layer):

  full_name = 'partition'
  abbreviation = 'patch'

  def __init__(self, patch_size):
    self.patch_size = patch_size
    self.dimension = None


  @property
  def structure_tail(self):
    return '({})'.format(self.patch_size)


  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    # ATTENTION 1: square patches are used
    # ATTENTION 2: pixels fell outside the grids are ignored

    x_shape = x.shape.as_list()
    self.dimension = len(x_shape) - 2
    assert self.dimension == 2

    n_patches = np.prod(x_shape[1:-1]) // (self.patch_size ** self.dimension)

    sizes = (1, self.patch_size, self.patch_size, 1)
    strides = (1, self.patch_size, self.patch_size, 1)
    rates = [1] * len(x_shape)

    y = tf.image.extract_patches(x, sizes, strides, rates, padding='VALID')
    shape = [-1, n_patches] + [self.patch_size] * self.dimension + [x_shape[-1]]
    y = tf.reshape(y, shape=shape)

    return y



if __name__ == '__main__':
  # This test snippet is created by LLM
  import matplotlib.pyplot as plt

  # 100x100 color block
  image = np.zeros((1, 100, 100, 3))
  for i in range(100):
    for j in range(100):
      image[0, i, j, 0] = i / 100
      image[0, i, j, 1] = j / 100
      image[0, i, j, 2] = (i + j) / 200

  # Create layer to test
  partition_layer = Partition(patch_size=10)

  # Create graph
  x = tf.placeholder(tf.float32, shape=[1, 100, 100, 3])
  y = partition_layer(x)
  print('y.shape =', y.shape.as_list())

  # Run session
  with tf.Session() as sess:
    patches = sess.run(y, feed_dict={x: image})

  # Show original image
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(image[0])
  plt.title('Original Image')

  # Show partitioned image
  plt.subplot(1, 2, 2)
  num_patches = patches.shape[1]
  rows = int(np.sqrt(num_patches))
  cols = rows
  patch_images = np.reshape(patches, [rows, cols, 10, 10, 3])

  # Define gap
  gap = 2
  # Plot patch images
  combined_image = np.ones(
    (rows * (10 + gap) - gap, cols * (10 + gap) - gap, 3))
  for i in range(rows):
    for j in range(cols):
      start_row = i * (10 + gap)
      start_col = j * (10 + gap)
      combined_image[start_row:start_row + 10, start_col:start_col + 10, :] = \
      patch_images[i, j]

  plt.imshow(combined_image)
  plt.title('Patched Image with Gap')

  plt.show()