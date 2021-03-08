tframe
===

A deep learning programming framework based on tensorflow, especially designed for neural network researchers who are devoted to developing new network architectures and are interested in analyzing network dynamics such as transition Jacobian $\frac{\mathbf{s}_t}{\mathbf{s}_{t-1}}$ in training RNNs.

---
## Installation
Currently this library has not been released to PyPI for it still has not been stable enough and modifications are made constantly. License will be added once this library is ready for release.

To use this library, just clone it to the working directory of your project.
The recommended version of tensorflow is 1.14.0.

---
## Content
- [Brief Introduction](#brief-introduction)
    - [Recommended Project Organization](#recommended-project-organization)
- Cool Stuff
    - Summary Viewer
- [Update News](#update-news)
    - [Hyperparameter Searching Engine is Now Available](#hyperparameter-searching-engine-is-now-available)

## Brief Introduction

Comming soon ...

### Recommended Project Organization

Comming soon ...

## Update News

### Hyperparameter Searching Engine is Now Available!

tframe users are familiar with the `script` which usually looks like this:

```Python
import sys
sys.path.append('../')

from cf10_core import th
from tframe.utils.script_helper import Helper


s = Helper()
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
s.register('gather_summ_name', s.default_summ_name + '.sum')
s.register('gpu_id', 0)

# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('epoch', 1000)

s.register('lr', 0.003, 0.001, 0.0003)
s.register('batch_size', 16, 32, 64, 128)
s.register('kernel_size', 3, 4, 5)
s.register('patience', 5, 10)
s.register('augmentation', s.true_and_false)

#s.configure_engine(strategy='skopt', criterion='Test accuracy')
s.run()
```
Running this directly will launch a grid-searching process for the hyperparameters specified in the above code.
The time needed for finding a good hyperparameter combination will be prohibitive when too many hyperparameters are involved. 
Thus a machine learning based hyperparameter seaching engine is needed. 
Following the designing philosophy of tframe, the usage of this engine is designed to be theoretically concisest:
based on the grid-search code, simplely add one line before `s.run()`:
```Python
s.configure_engine(strategy='skopt', criterion='Test accuracy')
```
This tolds the engine which strategy to use and which criterion to optimize.
The criterion should be one of the `Note.criteria.keys()` which will be shown in summary viewer.
By setting this, tframe will do the following thing:

- Automatically find the `.sum` file and read the results before deciding to explore next hyperparameter combinations.
- Automatically choose the hyperparamter type (categorical, integer or real) and the scale 
(uniform or log-uniform) based on tframe Flag type.
If `strategy` is `grid` which is the default, all hyperparameters will be regarded as being 
categorical. 
If any `strategy` capable of searching in a continuous subspace is specified,
 integer or real hyperparameters such as `learning_rate` and `batch_size` will be set to 
 their corresponding type.
- Automatically set the `low` and `high` values of each hyperparamter based on categorical 
choices given.
- Automatically read and filter the unseen `Notes` and feed them to the searching engine. 
- Wisely choose the engine configurations. E.g., if a configuration passed through the system arguments will overwrite the same setting specified in the script module.

With this engine, projects deployed on different machines can easily work parallelly once set 
to a same `summary path`.

