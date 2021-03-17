from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.layers import Activation
from tframe.layers import BatchNormalization
from tframe.layers import Conv2D
from tframe.layers import Dense
from tframe.layers import GlobalAveragePooling2D
from tframe.layers import MaxPool2D
from tframe.layers import Merge

from tframe.nets.forkmerge import ForkMergeDAG

from .conv_net import ConvNet


class NAS101(ConvNet):
  """This class provides a slightly larger ConvNet search space than that
  described in the NAS101 paper. Each architecture in the search space is
  specified by the input arguments of the constructor of this class.
  """

  CONV1X1 = 'conv1x1'  # usually followed by `-bn-relu`
  CONV3X3 = 'conv3x3'
  MAXPOOL3X3 = 'maxpool3x3'

  typical_vertices = 'conv3x3,maxpool3x3,conv3x3,conv1x1,conv3x3'
  typical_edges = '1;01;001;1000;10011;100001'

  def __init__(self, vertices=typical_vertices, edges=typical_edges,
               num_stacks=3, stem_channels=128, cells_per_stack=3,
               use_batchnorm=True, input_projection=True, **kwargs):
    self.vertices = vertices
    self.edges = edges
    # Make sure the graph is legal
    self._check_graph()
    self.adj_matrix = ForkMergeDAG.parse_edge_str(
      self.edges, len(self.vertices) + 2)

    self.num_stacks = num_stacks
    self.stem_channels = stem_channels
    self.use_batchnorm = use_batchnorm
    self.input_projection = input_projection
    self.cells_per_stack = cells_per_stack
    self.kwargs = kwargs


  def _get_cell(self, input_channels, output_channels):
    """Get cell according to the specification."""
    # Calculate channel number for each vertex
    n_channels = self.back_prop_channels(output_channels, self.adj_matrix)
    # Calculate in degree for each vertex
    in_degrees = np.sum(self.adj_matrix, axis=0)[1:]
    assert in_degrees[-1] > 1

    # Initialize vertices
    vertices = [self.parse_layer_string(s, filters, self.use_batchnorm)
                for s, filters in zip(self.vertices, n_channels)]

    # Add final merge layer as output vertex
    merge_kwargs = {'max_trim': 1}
    if self.adj_matrix[0, -1]:
      final_merge = [Merge.Sum(**merge_kwargs) if in_degrees[-1] == 2
                     else Merge.ConcatSum(**merge_kwargs)]
    else: final_merge = [Merge.Concat()]
    vertices.append(final_merge)

    # Add internal merge layers
    for vertex, in_degree in zip(vertices[:-1], in_degrees[:-1]):
      assert in_degree > 0 and isinstance(vertex, list)
      if in_degree == 1: continue
      vertex.insert(0, Merge.Sum())

    # Determine projection layer
    input_projections = [[] for _ in vertices]
    for j, layers in enumerate(vertices):
      if not self.adj_matrix[0, j + 1]: continue
      # Do not add projection unless it will cause building error
      if not self.input_projection and any([
        not isinstance(layers[0], Merge),
        input_channels == output_channels]): continue
      # Add input projection
      input_projections[j] = self.conv_bn_relu(
        n_channels[j], 1, self.use_batchnorm)

    return ForkMergeDAG(vertices, self.adj_matrix, input_projections)


  def _get_layers(self):
    """The architecture follows a common pattern used in related literatures,
    that is, having a stem followed by stacks of cells. The image-like
    tensors will be down sampled after each stack. Finally a global average
    pooling layer is added."""
    layers = []
    # Add stem
    layers.extend(self.conv_bn_relu(self.stem_channels, 3))
    # Add stacks
    output_channels = self.stem_channels
    for i in range(self.num_stacks):
      input_channels = output_channels
      # Down sample except in first cell
      if i > 0:
        layers.append(MaxPool2D(2, 2))
        # Double channel number
        output_channels *= 2

      for _ in range(self.cells_per_stack):
        layers.append(self._get_cell(input_channels, output_channels))
        input_channels = output_channels

    # Add global average pooling layer
    layers.append(GlobalAveragePooling2D())

    return layers


  def _check_graph(self):
    """Do basic check for graph. Note that no prune procedure is involved.
    If the input graph contains extraneous vertices, error will be raised in
    ForkMergeDAG instance. """
    # Parse vertices if necessary
    if isinstance(self.vertices, str):
      self.vertices = self.vertices.split(',')
    # :: Check edges
    assert isinstance(self.edges, str)
    forward_specs = self.edges.split(';')
    # (1) Check specs length
    assert len(forward_specs) == len(self.vertices) + 1
    # (2) Check spec length for each vertex
    for i, spec in enumerate(forward_specs):
      # spec should consist of '0' and '1' and each vertex should be fed
      #  to at least one following vertex
      assert all([c in '01' for c in spec]) and '1' in spec
      assert len(spec) == i + 1


  @staticmethod
  def back_prop_channels(n_output_channel, matrix):
    """Calculate each channel number for each internal vertex according to
    NAS-101 paper."""
    assert isinstance(matrix, np.ndarray) and len(matrix.shape) == 2
    n_vertices = matrix.shape[0] - 2
    assert n_vertices > 0

    n_channels = [0] * n_vertices
    n_branches = np.sum(matrix[1:, -1])
    n_internal_channel = n_output_channel // n_branches
    n_channel_remain = n_output_channel % n_branches

    # Set channel number for vertices adjacent to output vertex
    back_indices = np.argwhere(matrix[1:, -1]).ravel()
    for i in back_indices:
      n_channels[i] = n_internal_channel
      if n_channel_remain:
        n_channels[i] += 1
        n_channel_remain -= 1

    # Set channel number for all other vertices
    other_indices = [i for i in range(n_vertices) if i not in back_indices]
    for i in reversed(other_indices):
      n_channels[i] = np.max(
        [n_channels[j] for j in np.argwhere(matrix[i + 1, 1:]).ravel()])
      assert n_channels[i] > 0

    # Sanity check and return
    n_channels.append(n_output_channel)
    return n_channels


  class Shelf(object):
    class CIFAR10(object):
      NAS101Best = ('conv3x3,maxpool3x3,conv3x3,conv1x1,conv3x3',
                    '1;01;001;1000;10011;100001')






