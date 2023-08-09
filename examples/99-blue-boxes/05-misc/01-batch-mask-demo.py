from tframe import Classifier, DataSet, mu, console
from tframe.utils.misc import convert_to_one_hot
from tframe import hub as th

import numpy as np



th.use_batch_mask = True

# (1) Prepare data
batch_size = 10
features = np.zeros(shape=[batch_size, 1])
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
targets = convert_to_one_hot(labels, 2)
batch = DataSet(features, targets, NUM_CLASSES=2)


# (2) Create a model
model = Classifier('foobar')
model.add(mu.Input([1]))
model.add(mu.Dense(2))
model.add(mu.Activation('softmax'))

model.build(metric=['accuracy'])

# (3) Evaluate model
console.section('Mask = *')
batch.data_dict['batch_mask'] = [1] * 10
results = model.classify(batch)
console.show_status(f'results = {results}')
model.evaluate_model(batch)
model.evaluate_pro(batch, show_class_detail=True, show_confusion_matrix=True)

console.section('Mask = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]')
batch.data_dict['batch_mask'] = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
results = model.classify(batch)
console.show_status(f'results = {results}')
model.evaluate_model(batch)
model.evaluate_pro(batch, show_class_detail=True, show_confusion_matrix=True)

