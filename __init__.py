import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string("mark", "default", "...")

flags.DEFINE_integer("epoch", -1, "Epochs to train")
flags.DEFINE_integer("batch_size", -1, "The size of batch images")

flags.DEFINE_bool("overwrite", True, "Whether to overwrite records")
flags.DEFINE_bool("shuffle", False, "Whether to shuffle")

FLAGS = flags.FLAGS


from .utils import console
from .utils import local
from .utils.tfdata import TFData

from .models import Predictor
from .models import GAN

from . import config
