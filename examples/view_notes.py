import sys, os
FILE_PATH = os.path.abspath(__file__)
DIR_DEPTH = 2
ROOT = FILE_PATH
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe.utils.summary_viewer.main_frame import SummaryViewer
from tframe import local
from tframe.utils.tensor_viewer.plugins import lottery
from tframe.utils.tensor_viewer.plugins import activation_sparsity


default_inactive_flags = (
  'patience',
  'shuffle',
  'epoch',
  'early_stop',
  'warm_up_thres',
  'mark',
  'save_mode',
  'output_size',
  'total_params',
  'dense_total_params',
  'pruning_rate_fc',
  'weights_fraction',
  'num_units',
)

flags_to_ignore = (
  'patience',
  'shuffle',
  'warm_up_thres',
  'epoch',
  'early_stop',
  'num_steps',
)

default_inactive_criteria = (
  'Mean Record',
  'Record',
)


while True:
  try:
    summ_path = local.wizard(extension='sum', max_depth=3,
                             current_dir=os.path.dirname(FILE_PATH),
                             # input_with_enter=True)
                             input_with_enter=False)

    if summ_path is None:
      input()
      continue
    print('>> Loading notes, please wait ...')
    viewer = SummaryViewer(
      summ_path,
      default_inactive_flags=default_inactive_flags,
      default_inactive_criteria=default_inactive_criteria,
      flags_to_ignore=flags_to_ignore,
    )
    # viewer.register_plugin(lottery.plugin)
    viewer.register_plugin(activation_sparsity.plugin)
    viewer.show()

  except Exception as e:
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
    input('Press any key to quit ...')
    raise e
  print('-' * 80)
