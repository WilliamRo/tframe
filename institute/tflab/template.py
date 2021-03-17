import numpy as np
# Put this line before importing tensorflow to get rid of future warnings
from tframe import console
import tensorflow as tf


console.suppress_logging()
console.start('Template')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
x = np.arange(12).reshape(6, 2)

tensor_x = tf.placeholder(dtype=tf.float32)
console.eval_show(tensor_x, name='x', feed_dict={tensor_x: x})
# =============================================================================
# End of the script
# =============================================================================
console.end()
