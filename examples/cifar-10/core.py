import sys, os
absolute_path = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# For example, if absolute_path is 'ROOT/tframe/examples/some_example/core.py',
#   then the DIR_DEPTH is 3.
DIR_DEPTH = 3
# Insert `ROOT/tframe/examples/some_example`, `ROOT/tframe/examples`,
#   `ROOT/tframe` and `ROOT` to sys.path.
ROOT = absolute_path
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
# Do some importing
from tframe import console, SaveMode
from tframe.trainers import TrainerHub
from data_utils import load_data
from tframe import Classifier


from_root = lambda path: os.path.join(ROOT, path)
# Create a hub and specify the directories
th = TrainerHub(as_global=True)
th.data_dir = from_root('tframe/examples/cifar-10/data/')
th.job_dir = from_root('tframe/examples/cifar-10/records')
# Specify your data shapes
th.input_shape = [32, 32, 3]
th.num_classes = 10
# Specify your device properties
th.allow_growth = False
th.gpu_memory_fraction = 0.35
# Specify some trainer stuff
th.save_mode = SaveMode.ON_RECORD
th.early_stop = True
th.patience = 5


def activate(export_false=False):
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Load data
  train_set, val_set, test_set = load_data(th.data_dir)
  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    model.evaluate_model(train_set)
    model.evaluate_model(val_set)
    model.evaluate_model(test_set, export_false=export_false)

  # End
  console.end()
