import tensorflow as tf
import core
from tframe import console
import model_lib as models


def main(_):
  console.start('LSTM task')

  # Configurations
  th = core.th
  th.model = models.lstm
  th.num_blocks = 1
  th.memory_depth = 2
  th.hidden_dim = 100

  th.epoch = 50000
  th.learning_rate = 1e-4
  th.batch_size = 8
  th.num_steps = 100
  th.val_preheat = 500
  th.validation_per_round = 5
  th.print_cycle = 2

  th.train = False
  th.smart_train = True
  th.max_bad_apples = 4
  th.lr_decay = 0.5

  th.save_model = True
  th.overwrite = True
  th.export_note = True
  th.summary = True
  th.monitor = False

  description = ''
  th.mark = '{}x{}{}'.format(th.num_blocks, th.memory_depth, description)

  core.activate()


if __name__ == '__main__':
  tf.app.run()



