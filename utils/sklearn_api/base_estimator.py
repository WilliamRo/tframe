from collections import OrderedDict
from roma import Nomear
from sklearn.utils.validation import check_X_y, check_array
from tframe import tf

import numpy as np



class BaseEstimator(Nomear):
  """Base class for all estimators in tframe"""
  def __init__(self, lr=0.01, patience=10, max_iterations=1e9,
               optimizer='sgd', tol=1e-3, **hp):
    self.lr = lr
    self.patience = patience
    self.tol = tol
    self.max_iterations = int(max_iterations)
    self.optimizer = optimizer

    self.hp_dict = hp

    # Private attributes
    self._input_shape = None
    self._is_fitted = False

    self._np_variables = OrderedDict()

  # region: Properties

  @Nomear.property(local=True)
  def tf_X(self):
    assert isinstance(self._input_shape, list)
    return tf.placeholder(tf.float32, [None] + self._input_shape, name='X')

  @Nomear.property(local=True)
  def tf_y(self):
    return tf.placeholder(tf.float32, [None, 1], name='y')

  @Nomear.property(local=True)
  def tf_var_dict(self) -> dict:
    assert len(self._np_variables) > 0

    d = OrderedDict()
    for k, v in self._np_variables.items():
      assert isinstance(v, np.ndarray)
      if k.lower() in ('b', 'bias'): initializer = tf.zeros_initializer()
      else: initializer = tf.random_normal_initializer()
      d[k] = tf.get_variable(k, shape=v.shape, initializer=initializer)

    return d

  @Nomear.property(local=True)
  def pred_tensor(self): return self._predict_tf(self.tf_X)

  @property
  def loss_tensor(self):
    return self._loss_function_tf(self.tf_y, self.pred_tensor)

  @property
  def score_tensor(self):
    return self._score_function_tf(self.tf_y, self.pred_tensor)

  @property
  def update_op(self):
    if self.optimizer.lower() in ('sgd',):
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
    else: raise ValueError(f'Unsupported optimizer `{self.optimizer}`')

    return optimizer.minimize(self.loss_tensor)

  # endregion: Properties

  # region: Public methods

  def fit(self, X: np.ndarray, y: np.ndarray):
    # (0) Check input
    X, y = check_X_y(X, y)
    self._input_shape = list(X.shape[1:])
    # The shape of y should be [?, 1], otherwise it will be broadcast
    #  to [?, ?] when calculating loss, causing unexpected results
    feed_dict = {self.tf_X: X, self.tf_y: y.reshape([-1, 1])}

    # (1) Initialize variables
    if len(self._np_variables) == 0: self._init_np_variables(X)

    # (2)
    with tf.Session() as sess:
      # Initialize variables
      patience = self.patience
      best_loss = np.inf
      best_var_dict = {}

      _ = self.tf_var_dict
      update_op, loss_tensor = self.update_op, self.loss_tensor

      # Initialize variables
      sess.run(tf.global_variables_initializer())

      # Iteratively train the model
      for i in range(self.max_iterations):
        if patience < 0: break

        # Run train step
        _, loss = sess.run([update_op, loss_tensor], feed_dict=feed_dict)

        if np.isnan(loss): break

        if loss < best_loss - self.tol:
          best_loss = loss
          patience = self.patience
          for k, tensor in self.tf_var_dict.items():
            best_var_dict[k] = sess.run(tensor)
        else:
          patience -= 1

    # (-1) Assign values and finalize
    for k, v in best_var_dict.items(): self._np_variables[k] = v
    self._is_fitted = True

  def predict(self, X):
    assert self._is_fitted
    X = check_array(X)
    y: np.ndarray = self._predict_np(X)
    return y.squeeze(-1)

  def predict_proba(self, X):
    assert self._is_fitted
    X = check_array(X)
    return self._predict_proba_np(X)

  # endregion: Public methods

  # region: Abstract methods

  def _init_np_variables(self, X: np.ndarray):
    raise NotImplementedError

  def _predict_np(self, X: np.ndarray):
    raise NotImplementedError

  def _predict_tf(self, X: tf.Tensor):
    raise NotImplementedError

  def _predict_proba_np(self, X):
    raise NotImplementedError

  def _loss_function_tf(self, y_true, y_pred):
    raise NotImplementedError

  def _score_function_tf(self, y_true, y_pred):
    return -self._loss_function_tf(y_true, y_pred)

  # endregion: Abstract methods

