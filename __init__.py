import tensorflow as tf

# Before importing other tframe modules, define FLAGS
flags = tf.app.flags

flags.DEFINE_string("mark", "default", "...")

flags.DEFINE_bool("act_sum", False, "Whether to add activation summaries")

flags.DEFINE_integer("epoch", -1, "Epochs to train")
flags.DEFINE_integer("batch_size", -1, "The size of batch images")

flags.DEFINE_integer("print_cycle", -1, "Print cycle")
flags.DEFINE_integer("snapshot_cycle", -1, "Snapshot cycle")
flags.DEFINE_integer("match_cycle", -1, "Match cycle for RL")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite records")
flags.DEFINE_bool("shuffle", False, "Whether to shuffle")

flags.DEFINE_bool("suppress_logging", True, "...")

flags.DEFINE_bool("train", True, "Whether to train or inference")

FLAGS = flags.FLAGS

from .core import with_graph

from .utils import console
from .utils import local
from .utils.tfdata import TFData
from .utils.tfinter import ImageViewer

from .models import Classifier
from .models import Predictor
from .models import GAN
from .models import VAE
from .models import TDPlayer

from . import config
from . import pedia


# Control logging
console.set_logging_level(1)

# Record graph bound to the last initiated model
current_graph = None

