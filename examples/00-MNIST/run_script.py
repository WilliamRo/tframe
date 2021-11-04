"""
This script is used for launching tframe script on a GPU server.

SYNTAX: python run_script.py n xy ...

:params n: Model ID. -> integer 2 digits
:params xy: script_suffix. x is GPU ID, y is the index of process on GPU x.
:params z: GPU memory fraction.
"""
import os, sys
sys.path.insert(0, '../roma')
import subprocess

from roma import console
from roma import finder


on_linux = os.name == 'posix'

DEFAULT_MEMORY_FRACTION = 0.3
PYTHON = 'python3.7' if on_linux else 'python'
GPU_NUM = 2

configs = sys.argv[1:]
n = configs[0]
xy = configs[1]

# Find script path
ptn = '{:02}_*'.format(int(n))
candidates = finder.walk(
  os.path.dirname(os.path.abspath(__file__)), type_filter='dir', pattern=ptn)
assert len(candidates) == 1
folder_path = candidates[0]

candidates = finder.walk(folder_path, type_filter='file', pattern='s*_*.py')
assert len(candidates) == 1
script_path = candidates[0]

# Find GPU_ID
assert len(xy) == 2
gpu_id = xy[0]
assert 0 <= int(gpu_id) < GPU_NUM

# Run
cmd = [PYTHON, script_path]
cmd.append('--gpu_id={}'.format(gpu_id))
cmd.append('--script_suffix={}'.format(xy))
cmd.append('--allow_growth=False')
cmd += configs[2:]
if on_linux: cmd.append('--python_version=3.7')

console.show_status('Executing `{}` ...'.format(' '.join(cmd)), color='yellow')
subprocess.run(cmd)

