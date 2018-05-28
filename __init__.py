from . import pedia
from .enums import *

from .config import Config
# Register
Config.register()
hub = Config()

from .utils import checker
from .utils import console
from .utils import local
from tframe.data.dataset import DataSet

from .import models

from .models import Predictor
from .models import Classifier

from .import core

from .monitor import Monitor
monitor = Monitor()

# TODO:
trainer = None


# Record graph bound to the last initiated model
# TODO
current_graph = None

