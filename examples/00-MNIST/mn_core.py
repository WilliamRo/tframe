import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 3
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe import console, SaveMode
from tframe.trainers import SmartTrainerHub
from tframe import Classifier
from tframe import monitor

import mn_ad as ad
import mn_du as du


from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = SmartTrainerHub(as_global=True)
th.data_dir = from_root('tframe/examples/00-MNIST/data/')
th.job_dir = from_root('tframe/examples/00-MNIST')
# -----------------------------------------------------------------------------
# Device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30
# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [28, 28, 1]
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
th.export_note = False
th.gather_note = True

th.print_cycle = 5
th.validation_per_round = 20
th.export_tensors_upon_validation = True
th.sample_num = 3

th.val_batch_size = 1000
th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True


def activate(export_false=False):
  # Register activation filter
  if th.export_activations:
    monitor.register_activation_filter(ad.act_type_ii_filter)

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
    bs = 5000
    model.evaluate_model(train_set, batch_size=bs)
    if not th.test_directly: model.evaluate_model(val_set, batch_size=bs)
    model.evaluate_model(test_set, export_false=export_false, batch_size=bs)

  # End
  model.shutdown()
  console.end()
