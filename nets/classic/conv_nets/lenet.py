from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.layers.advanced import Dense
from tframe.layers.convolutional import Conv2D
from tframe.layers.common import Flatten, Dropout

from .conv_net import ConvNet


class LeNet(ConvNet):

  def __init__(self, archi_string='6-16=120-84', kernel_size=5,
               strides=2, padding='valid', activation='tanh', dropout=0.0):
    self.conv_list, self.fc_list = self.parse_archi_string(archi_string)
    self.kernel_size = kernel_size
    self.padding = padding
    self.strides = strides
    self.activation = activation
    self.dropout = dropout


  def _get_layers(self):
    layers = []
    # Add conv layers
    for n in self.conv_list: layers.append(Conv2D(
      n, self.kernel_size, strides=self.strides,
      padding=self.padding, activation=self.activation))
    # Add flatten layer
    layers.append(Flatten())
    # Add fully-connected layers
    for n in self.fc_list:
      layers.append(Dense(n ,activation=self.activation))
      if self.dropout > 0:
        assert self.dropout < 1
        layers.append(Dropout(train_keep_prob=1 - self.dropout))
    return layers


