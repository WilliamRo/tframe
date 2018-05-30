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
from data_utils import load_data
from tframe import Predictor


from_root = lambda path: os.path.join(ROOT, path)

th = SmartTrainerHub(as_global=True)
th.data_dir = from_root('tframe/examples/whbm/data/')
th.job_dir = from_root('tframe/examples/whbm')

th.allow_growth = False
th.gpu_memory_fraction = 0.4

th.save_mode = SaveMode.ON_RECORD
th.warm_up_thres = 1
th.at_most_save_once_per_round = True

th.early_stop = True
th.idle_tol = 10


def activate():
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Predictor)

  # Load data
  train_set, val_set, test_set = load_data(th.data_dir, th.memory_depth)

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    model.evaluate_model(train_set)
    model.evaluate_model(val_set)
    model.evaluate_model(test_set)

  # End
  console.end()
