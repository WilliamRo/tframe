import tensorflow as tf
import misc, model_lib
from data_utils import load_data

from tframe import hub, console, SaveMode
from tframe.trainers import SmartTrainerHub

hub.data_dir = misc.from_root('specify your data path here')


def main(_):
  console.start('Example task')

  # Configurations
  th = SmartTrainerHub(as_global=True)
  # ...

  th.epoch = 10
  th.learning_rate = 1e-4
  th.batch_size = 64
  th.validation_per_round = 10
  th.print_cycle = 100

  # th.train = False
  # th.smart_train = True
  th.max_bad_apples = 4
  th.lr_decay = 0.6

  th.early_stop = True
  th.idle_tol = 20
  th.save_mode = SaveMode.ON_RECORD
  th.warm_up_thres = 1
  th.at_most_save_once_per_round = True

  th.save_model = False
  th.overwrite = True
  th.export_note = True
  th.summary = False
  th.monitor = False

  th.allow_growth = False
  th.gpu_memory_fraction = 0.4

  description = '0'
  th.mark = 'example_model_{}'.format(description)

  # Fetch your model from lib
  model = model_lib.example_model(th)

  # Load data
  train_set, val_set, test_set = load_data(th.data_dir)

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    pass

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()



