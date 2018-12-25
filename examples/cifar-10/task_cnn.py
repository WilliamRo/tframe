import tensorflow as tf
import core
from tframe import console
from tframe.utils.misc import mark_str as ms
import model_lib as models


def main(_):
  console.start('CNN task on CIFAR-10')

  # Configurations
  th = core.th
  th.model = models.cnn
  th.suffix = '_x'
  th.job_dir = 'records_cnn'
  th.gather_summ_name = 'cnn.sum'
  th.fc_dims = [32, 64]
  th.actype1 = 'relu'
  th.use_batchnorm = True

  th.epoch = 50
  th.learning_rate = 1e-4
  th.optimizer = tf.train.AdamOptimizer(th.learning_rate)
  th.batch_size = 64
  th.validation_per_round = 4
  th.val_batch_size = 512
  th.print_cycle = 10
  th.shuffle = True

  th.train = True

  th.save_model = False
  th.overwrite = True
  th.summary = False
  th.export_note = True
  th.gather_note = True
  th.validate_train_set = True
  th.export_kernel = True
  th.export_tensors_upon_validation = True

  th.show_structure_detail = True
  description = ''
  th.mark = 'cnn_{}{}'.format(ms(th.fc_dims), description)

  export_false = True
  core.activate(export_false=export_false)


if __name__ == '__main__':
  tf.app.run()



