from . import pedia
from .enums import *

from .utils import console
from .utils import local
from .utils.tfdata import TFData
from .utils.tfinter import ImageViewer

from .config import Config
from .import models

from .models import Predictor
from .models import Classifier

from . import core


# Register
Config.register()
config = Config()

# Record graph bound to the last initiated model
# TODO
current_graph = None

