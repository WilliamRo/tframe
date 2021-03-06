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
from tframe.trainers import SmartTrainerHub
from tframe import Classifier

import cf10_du as du


from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = SmartTrainerHub(as_global=True)
th.data_dir = from_root('tframe/examples/01-CIFAR10/data/')
th.job_dir = from_root('tframe/examples/01-CIFAR10')
# -----------------------------------------------------------------------------
# Device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30
# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [32, 32, 3]
th.num_classes = 10

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.show_structure_detail = True
th.keep_trainer_log = True
th.warm_up_thres = 1

th.early_stop = True
th.patience = 6
th.shuffle = True

th.save_model = False
th.save_mode = SaveMode.ON_RECORD
th.overwrite = True
th.summary = False
th.export_note = False
th.gather_note = True

th.print_cycle = 5
th.validation_per_round = 20
th.export_tensors_upon_validation = True
th.sample_num = 3

th.centralize_data = False
th.val_batch_size = 2000

th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True


def activate(export_false=False):
  # Load data
  train_set, val_set, test_set = du.load_data(th.data_dir)
  if th.centralize_data: th.data_mean = train_set.feature_mean

  # Build model
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th,
                test_set=test_set)
  else:
    model.evaluate_model(train_set, batch_size=1000)
    model.evaluate_model(val_set, batch_size=1000)
    model.evaluate_model(test_set, export_false=export_false, batch_size=1000)

  # End
  model.shutdown()
  console.end()
