from .core import with_graph

from .utils import console
from .utils import local
from .utils.tfdata import TFData
from .utils.tfinter import ImageViewer

from .models import Predictor
from .models import Classifier
from .models import GAN
from .models import VAE

from . import pedia

from .config import Config
Config.register()
config = Config()

# Record graph bound to the last initiated model
# TODO
current_graph = None

