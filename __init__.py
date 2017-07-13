import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_integer("epoch", -1, "Epochs to train")
flags.DEFINE_integer("batch_size", -1, "The size of batch images")

FLAGS = flags.FLAGS


from .utils import console
from .utils import local
from .utils.tfdata import TFData

from .models import Feedforward
