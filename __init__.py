import tensorflow as tf

# Before importing other tframe modules, define FLAGS
flags = tf.app.flags

flags.DEFINE_string("mark", "default", "...")

flags.DEFINE_bool("act_sum", False, "Whether to add activation summaries")

flags.DEFINE_integer("epoch", -1, "Epochs to train")
flags.DEFINE_integer("batch_size", -1, "The size of batch images")

flags.DEFINE_integer("print_cycle", -1, "Print cycle")
flags.DEFINE_integer("snapshot_cycle", -1, "Snapshot cycle")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite records")
flags.DEFINE_bool("shuffle", False, "Whether to shuffle")

flags.DEFINE_bool("suppress_logging", True, "...")

flags.DEFINE_bool("train", True, "Whether to train or inference")

FLAGS = flags.FLAGS

from .utils import console
from .utils import local
from .utils.tfdata import TFData

from .models import Classifier
from .models import Predictor
from .models import GAN

from . import config
from . import pedia


# Control logging
console.set_logging_level(1)

# Create placeholder for tframe status
is_training = tf.placeholder(dtype=tf.bool, name=pedia.is_training)
tf.add_to_collection(pedia.status_tensors, is_training)
pedia.memo[pedia.is_training] = is_training
