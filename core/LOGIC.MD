## Content

- [Exporting Tensors](#exporting-tensors)

  - [Exporting weight gradients](#exporting-weight-gradients)

  - [Exporting neuron activations](#exporting-neuron-activations)

    - [Type I activation](#type-i-activation)
    - [Type II activation](#type-ii-activation)

    

## Exporting Tensors

There are 2 types of Tensors:

- **Type I**: tf.Variables` that can be fetched without `feed_dict`, such as trainable parameters and static weight masks. Numpy arrays that stored in `Monitor` also belong to this type.
- **Type II**: `tf.Tensors` that can not be fetched without feeding data samples, such as neuron activations and gradients.

Tensors will be exported to `Note` during training. Specifically, this is done in `Trainer._inner_loop -> self._take_notes_for_export() `. When this method takes effect, `scalars`, such as `batch_loss` and `val_accuracy` will be taken down in `Note` along with `tensors`. After training, `Note` will be saved individually or into a summary file that is solely a list of notes. `scalars` and `tensors` can be displayed using `tframe.utils.tensor_viewer`:

![img](https://github.com/WilliamRo/imgs/blob/master/tensor_viewer_1.png?raw=true)

Currently logic for exporting both types of `tensors` at the same time is not optimal since it's tricky to design the corresponding drop-down list for selecting tensors (bottom right widget in the snapshot above).

### Exporting weight gradients

Currently only exporting Type I weight gradients for *feedforward nets* is supported, which means only the running average of weight gradients can be exported.

To export weight gradient, set

```python
# When this option is set to True, smooth_out_conflicts method in config_base.py will
#   set th.monitor_weight_grads to True, which needs to be optimized.
th.export_weight_grads = True
```

With this, all learnable weights will be registered to `context.monitor` by calling `Net.variable_extractor() -> _register_weights_to_monitor()` method at the end of `_link` method in `Net` or `RNet`. In `Monitor.register_weights` method, an instance of `Statistic` will be created for each weight variable. (Step 1)

During building model (in `Predictor._build`), `loss` tensor will be registered to monitor to create `Monitor._grad_ops` that will be plugged into `Model.grads_slot` and added to `Model._update_group` later. (Step 2)

During training, in each update step, gradients will be recorded to `context.monitor` in `Trainer._update_model` method. At this point, the moving average of weight gradients are ready to be taken down to `Note`. (Step 3)

In `Trainer._inner_loop -> _take_notes_for_export() -> _get_tensors_to_export() -> _get_variables_to_export()`:

```python
if self.th.export_weight_grads:
  for key, value in context.monitor.grad_dict.items(): _add_to_dict(key, value)
```

Then `tensors` will be taken down in `Note`. (Step 4)

### Exporting neuron activations

Both Type I and Type II neuron activations for *feedforward nets* can be exported. Like exporting weight gradients, the first step to export neuron activations is to turn on the monitor option:

```python
th.monitor_activations = True
```

For distinction between Type I and Type II, an `activation_filter` (there can only be one) should be registered to `monitor`. An `activation_filter` takes a `tf.Tensor` as input, and returns an integer (often based on the tensor name) to `Monitor.register_tensor` method (this method is only used inside `Net._link` and `RNet._link` method). The integer can be: 

- 0: the corresponding tensor will be ignored
- 1: the corresponding tensor will be registered as Type I
- 2: the corresponding tensor will be registered as Type II

#### Type I activation

A corresponding `Statistic` will be created and registered `monitor._tensor_stats_dict`. Like exporting Type I weight gradients, in `Predictor._build` method, `Model.general_tensor_slot` will be plugged in with `monitor.tensor_fetches` which will be added to `Model.update_group`. During each training step, the  Type I activation `Statistics` will be updated. In each note cycle, these stats will be taken down to `Note`.

#### Type II activation

The input `Tensor` will be added to `context.tensors_to_export` dictionary. In each note cycle, these tensors will be fetched with `th.sample_num` samples being fed into `model`.



















