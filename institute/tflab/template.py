import numpy as np
import tensorflow as tf
from tframe import console


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
