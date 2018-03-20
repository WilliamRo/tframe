import tensorflow as tf

# Before importing other tframe modules, define FLAGS
flags = tf.app.flags

flags.DEFINE_string("mark", "default", "...")

flags.DEFINE_bool("act_sum", False, "Whether to add activation summaries")

flags.DEFINE_integer("epoch", -1, "Epochs to train")
flags.DEFINE_integer("epoch_tol", 20, "epoch tolerance")
flags.DEFINE_integer("batch_size", -1, "The size of batch images")

flags.DEFINE_integer("print_cycle", -1, "Print cycle")
flags.DEFINE_integer("snapshot_cycle", -1, "Snapshot cycle")
flags.DEFINE_integer("match_cycle", -1, "Match cycle for RL")
flags.DEFINE_integer("dont_save_until", 1, "Until which do not save")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite records")
flags.DEFINE_bool("shuffle", False, "Whether to shuffle")

flags.DEFINE_bool("suppress_logging", True, "...")

flags.DEFINE_bool("train", True, "Whether to train or inference")
flags.DEFINE_bool("smart_train", False, "Whether to train in a smart way")
flags.DEFINE_bool("save_best", False, "Whether to save best model")
flags.DEFINE_bool("use_default", True, "Whether to use default setting")

flags.DEFINE_float("lr_decay", 0.6, "Learning rate decay ratio in smart train")

flags.DEFINE_string("job_dir", "./", "The root directory where the records "
                                     "should be put")
flags.DEFINE_string("data_dir", "", "The data directory")

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
# from .models import TDPlayer

from . import config
from . import pedia


# Control logging
console.set_logging_level(1)

# Record graph bound to the last initiated model
current_graph = None

