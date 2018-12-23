from tframe.utils.script_helper import Helper


module_name = 'task_mlp.py'
job_dir = 'mlp_records'
notes_name = 'mlp.sum'

s = Helper(module_name)

s.register('job-dir', job_dir)
s.register('gather_note', True)
s.register('gather_summ_name', notes_name)
s.register('gpu_id', 1)
s.register('epoch', 50)

true_and_false = (True, False)

s.register('lr', (0.01, 0.001))
s.register('batch_size', (32, 64))
s.register('use_batchnorm', False)

s.run(5)
