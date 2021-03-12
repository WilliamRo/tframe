from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Ignore FutureWarnings before import tensorflow
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import pedia
from .enums import *

from .import core
from .core.context import context
from .core.context import hub
from .core.context import monitor

from .utils import checker
from .utils import console
from .utils import local
from .utils import linker
from tframe.data.dataset import DataSet

from .import models

from .models import Predictor
from .models import Classifier

from .utils.organizer import mu

from .trainers.smartrainer import SmartTrainerHub as DefaultHub

import numpy as np
import tensorflow as tf


def set_random_seed(seed=26):
  np.random.seed(seed)
  tf.set_random_seed(seed)
