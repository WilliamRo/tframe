from . import pedia
from .enums import *

from .config import Config
# Register
Config.register()
hub = Config()

from .utils import console
from .utils import local
from .utils.tfdata import TFData
from .utils.tfinter import ImageViewer

from .import models

from .models import Predictor
from .models import Classifier

from . import core


# Record graph bound to the last initiated model
# TODO
current_graph = None

