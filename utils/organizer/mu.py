# ---------------------------------------------------------------
#  Model
# ---------------------------------------------------------------
from tframe.models.sl.classifier import Classifier
from tframe.models.sl.predictor import Predictor
from tframe.models import Feedforward
from tframe.models import Recurrent

# ---------------------------------------------------------------
#  Net
# ---------------------------------------------------------------
from tframe.nets.classic.conv_nets.lenet import LeNet
from tframe.nets.classic.conv_nets.nas101 import NAS101
from tframe.nets.classic.conv_nets.unet import UNet
from tframe.nets.classic.conv_nets.unet2d import UNet2D

from tframe.nets.forkmerge import ForkMerge
from tframe.nets.forkmerge import ForkMergeDAG

# ---------------------------------------------------------------
#  Layer
# ---------------------------------------------------------------
from tframe.layers.advanced import Dense

from tframe.layers.common import Activation
from tframe.layers.common import Dropout
from tframe.layers.common import Flatten
from tframe.layers.common import Input
from tframe.layers.common import Linear
from tframe.layers.common import Onehot
from tframe.layers.common import Reshape

from tframe.layers.convolutional import Conv1D
from tframe.layers.convolutional import Conv2D
from tframe.layers.convolutional import Deconv2D

from tframe.layers.highway import LinearHighway

from tframe.layers.hyper.conv import Conv1D as HyperConv1D
from tframe.layers.hyper.conv import Conv2D as HyperConv2D
from tframe.layers.hyper.conv import Conv3D as HyperConv3D
from tframe.layers.hyper.conv import Deconv1D as HyperDeconv1D
from tframe.layers.hyper.conv import Deconv2D as HyperDeconv2D
from tframe.layers.hyper.conv import Deconv3D as HyperDeconv3D
from tframe.layers.hyper.conv import DenseUpsampling2D
DUC = DenseUpsampling2D

from tframe.layers.hyper.dense import Dense as HyperDense

from tframe.layers.merge import Bridge
from tframe.layers.merge import Merge
from tframe.layers.merge import ShortCut

from tframe.layers.normalization import BatchNormalization
from tframe.layers.normalization import LayerNormalization

from tframe.layers.preprocess import Normalize

from tframe.layers.pooling import AveragePooling2D
from tframe.layers.pooling import GlobalAveragePooling1D
from tframe.layers.pooling import GlobalAveragePooling2D
from tframe.layers.pooling import MaxPool1D
from tframe.layers.pooling import MaxPool2D
from tframe.layers.pooling import MaxPool3D

# ---------------------------------------------------------------
#  Layer
# ---------------------------------------------------------------
from tframe.nets.rnn_cells.lstms import LSTM
from tframe.nets.rnn_cells.gdu import GDU
from tframe.nets.rnn_cells.gru import GRU

# ---------------------------------------------------------------
#  Alias
# ---------------------------------------------------------------
BatchNorm = BatchNormalization



