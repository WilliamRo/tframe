import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper


s = Helper()
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
s.register('epoch', 5)
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
gpu_id = 0
summ_name = s.default_summ_name

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
# -----------------------------------------------------------------------------
# Specified hyper-parameters
# -----------------------------------------------------------------------------
s.register('batch_size', 32, 64)

# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('gpu_memory_fraction', 0.3)
s.run(5, save=False)

