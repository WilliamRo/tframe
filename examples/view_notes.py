import sys, os
absolute_path = os.path.abspath(__file__)
DIR_DEPTH = 2
ROOT = absolute_path
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe.utils.summary_viewer.main_frame import SummaryViewer
from tframe import local


try:
  notes_path = local.wizard('sum', current_dir=os.path.dirname(__file__),
                            input_with_enter=False, max_depth=2)
  print('>> Loading notes, please wait ...')
except Exception as e:
  import sys, traceback
  traceback.print_exc(file=sys.stdout)
  input('Press any key to quit ...')
  raise e

viewer = SummaryViewer(
  notes_path,
  default_inactive_flags=(
    'epoch',
    'mark',
  ))
viewer.show()

