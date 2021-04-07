from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker

from tframe.layers.common import Activation
from tframe.layers.convolutional import Conv2D, Deconv2D
from tframe.layers.pooling import MaxPool2D
from tframe.layers.common import Dropout
from tframe.layers.merge import Concatenate

from tframe.nets.net import Net


class ClassicUnet(Net):

  def __init__(
      self,
      num_filter_list,
      num_classes,
      kernel_initializer='glorot_uniform',
      left_repeats=2,
      right_repeats=2,
      activation='relu',
      dropout_rate=0.5,
      name='unet',
      level=1,
      **kwargs):

    # Sanity  check
    assert isinstance(num_filter_list, (tuple, list)) and num_filter_list
    # Call parent's constructor
    # TODO: the level logic is not elegant
    super().__init__(name, level=level, **kwargs)
    # Specific attributes
    self.num_filter_list = num_filter_list
    self.num_classes = checker.check_positive_integer(num_classes)
    # self.kernel_sizes = kernel_sizes
    self.kernel_initializer = kernel_initializer
    self.activation = activation
    self.dropout_rate = checker.check_type(dropout_rate, float)
    self.left_repeats = checker.check_positive_integer(left_repeats)
    self.right_repeats = checker.check_positive_integer(right_repeats)
    # Add layers
    self._add_layers()


  @property
  def net_height(self): return len(self.num_filter_list)


  def _add_layers(self):

    # Construct left half
    layers_to_link = []
    for i, filters in enumerate(self.num_filter_list):
      # Add conv layer
      for _ in range(self.left_repeats):
        last_layer = self._add_conv(filters, kernel_size=3)
      # Add dropout layer if necessary
      if i + 1 in (self.net_height, self.net_height - 1) and self.dropout_rate:
        last_layer = self.add(Dropout(1 - self.dropout_rate))
      # Save to layer list for future linking if necessary
      if i + 1 != self.net_height:
        layers_to_link.append(last_layer)
        # Add pooling layer
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Construct right half
    for filters, pre_layer in zip(
        reversed(self.num_filter_list[:-1]), reversed(layers_to_link)):
      # Up-sampling
      self.add(Deconv2D(
        filters, kernel_size=2, strides=2, activation=self.activation,
        kernel_initializer=self.kernel_initializer))
      # Add Conv layer
      # self._add_conv(filters, kernel_size=2)
      # Merge
      self.add(Concatenate(pre_layer))
      # Add Conv layers
      for _ in range(self.right_repeats): self._add_conv(filters, kernel_size=3)

    # Add output layer
    if self.num_classes == 2:
      self._add_conv(2, kernel_size=3)
      self._add_conv(1, kernel_size=1, use_activation=False)
      self.add(Activation('sigmoid'))
    else:
      self._add_conv(self.num_classes, kernel_size=3)
      self._add_conv(self.num_classes, kernel_size=1, use_activation=False)
      self.add(Activation('softmax'))


  def _add_conv(self, filters, kernel_size, use_activation=True):
    activation = self.activation if use_activation else None
    last_layer = self.add(Conv2D(
      filters, kernel_size=kernel_size, activation=activation,
      kernel_initializer=self.kernel_initializer))
    return last_layer


if __name__ == '__main__':
  for i in reversed([1, 2, 3]): print(i)
