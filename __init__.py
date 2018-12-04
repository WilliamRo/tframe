from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import pedia
from .enums import *

from tframe.configs.config_base import Config

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

