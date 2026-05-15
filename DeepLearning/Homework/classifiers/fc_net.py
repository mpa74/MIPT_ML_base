import os

import numpy as np

from .layer_utils import *
from .layers import *


class FullyConnectedNet:
  """Class for a multi-layer fully connected neural network.

  Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
  and a softmax loss function. This will also implement dropout and batch/layer
  normalization as options. For a network with L layers, the architecture will be

  {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch/layer normalization and dropout are optional and the {...} block is
  repeated L - 1 times.

  Learnable parameters are stored in the self.params dictionary and will be learned
  using the Solver class.
  """

  def __init__(
    self,
    hidden_dims,
    input_dim=3 * 32 * 32,
    num_classes=10,
    dropout_keep_ratio=1,
    normalization=None,
    reg=0.0,
    weight_scale=1e-2,
    dtype=np.float32,
    seed=None,
  ):
    """Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
        If dropout_keep_ratio=1 then the network should not use dropout at all.
    - normalization: What type of normalization the network should use. Valid values
        are "batchnorm", "layernorm", or None for no normalization (the default).
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
        initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
        this datatype. float32 is faster but less accurate, so you should use
        float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers.
        This will make the dropout layers deteriminstic so we can gradient check the model.
    """
    self.normalization = normalization
    self.use_dropout = dropout_keep_ratio != 1
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution centered at 0 with standard       #
    # deviation equal to weight_scale. Biases should be initialized to zero.   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to ones and shift     #
    # parameters should be initialized to zeros.                               #
    ############################################################################
    #
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
      if seed is not None:
        self.dropout_param["seed"] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []

    if self.normalization == "batchnorm":
      self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]

    if self.normalization == "layernorm":
      self.bn_params = [{} for i in range(self.num_layers - 1)]

    # Cast all parameters to the correct datatype.
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """Compute loss and gradient for the fully connected net.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
        scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.
    """
    X = X.astype(self.dtype)
    mode = "test" if y is None else "train"

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
      self.dropout_param["mode"] = mode
    if self.normalization == "batchnorm":
      for bn_param in self.bn_params:
        bn_param["mode"] = mode
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    #
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early.
    if mode == "test":
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch/layer normalization, you don't need to regularize the   #
    # scale and shift parameters.                                              #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class TwoLayerNet:
  """
      Двухслойная полносвязная нейронная сеть с нелинейностью ReLU и функцией потерь softmax,
      использующая модульную структуру слоев. Мы предполагаем размерность входных данных D,
      размерность скрытого слоя H и выполняем классификацию по C классам.

  Архитектура должна быть полносвязный слой - reLU - полносвязный слой - softmax.

  Обратите внимание, что этот класс не реализует градиентный спуск; вместо этого он
  будет взаимодействовать с отдельным объектом Solver, который отвечает за выполнение
  оптимизации.

  Обучаемые параметры модели хранятся в словаре
  self.params, который сопоставляет имена параметров с массивами numpy.
  """

  def __init__(
    self,
    input_dim=3 * 32 * 32,
    hidden_dim=100,
    num_classes=10,
    weight_scale=1e-3,
    reg=0.0,
  ):
    """
            Инициализация.

            Входы:
            - input_dim: Целое число, указывающее размер входных данных
    - hidden_dim: Целое число, указывающее размер скрытого слоя
    - num_classes: Целое число, указывающее количество классов для классификации
    - weight_scale: Скаляр, указывающий стандартное отклонение для случайной инициализации весов.
    - reg: Скаляр, указывающий силу L2-регуляризации.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Инициализировать веса и смещения двухслойной сети. Веса
    # должны быть инициализированы гауссовым распределением с центром в точке 0,0 и
    # стандартным отклонением, равным weight_scale, а смещения должны быть
    # инициализированы нулем. Все веса и смещения должны храниться в
    # словаре self.params, при этом веса первого слоя
    # и смещения будут использовать ключи 'W1' и 'b1', а веса и смещения второго слоя
    # будут использовать ключи 'W2' и 'b2'.

    ############################################################################

    self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params["b1"] = np.zeros(hidden_dim)
    self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b2"] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  def loss(self, X, y=None):
    """
            Вычисляет функцию потерь и градиент для мини-батча данных.
    Входные данные:
    - X: Массив входных данных формы (N, d_1, ..., d_k)
    - y: Массив меток формы (N, ..., d_k). y[i] — метка для X[i].

    Возвращает:
    Если y равно None, то выполняется прямой проход модели во время тестирования и возвращается:
    - scores: Массив формы (N, C), содержащий оценки классификации, где
    scores[i, c] — оценка классификации для X[i] и класса c.
    Если y не равно None, то выполняется прямой и обратный проходы во время обучения и
    возвращается кортеж из:
    - loss: Скалярное значение, определяющее функцию потерь
    - grads: Словарь с теми же ключами, что и self.params, сопоставляющий имена параметров
    с градиентами функции потерь относительно этих параметров.
    """
    scores = None
    ############################################################################
    # TODO: Реализовать прямой проход для двухслойной сети, вычисляя
    # оценки классов для X и сохраняя их в переменной scores.
    ############################################################################

    W1, b1 = self.params["W1"], self.params["b1"]
    W2, b2 = self.params["W2"], self.params["b2"]

    # Прямой проход
    h1 = X @ W1 + b1
    a1 = np.maximum(0, h1)  # ReLU
    scores = a1 @ W2 + b2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Реализовать обратный проход для двухслойной сети. Сохранить функцию потерь
    # в переменной loss, а градиенты — в словаре grads. Вычислить потери данных
    # с помощью softmax и убедиться, что grads[k] содержит градиенты для
    # self.params[k]. Не забудьте добавить L2-регуляризацию!
    # #
    # ПРИМЕЧАНИЕ: Чтобы убедиться, что ваша реализация соответствует нашей и вы проходите #
    # автоматизированные тесты, убедитесь, что ваша L2-регуляризация включает множитель #
    # равный 0,5 для упрощения выражения для градиента. #
    ############################################################################

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = X.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

    # Обратный проход
    dout = probs
    dout[np.arange(N), y] -= 1
    dout /= N

    grads["W2"] = a1.T @ dout + self.reg * W2
    grads["b2"] = np.sum(dout, axis=0)

    da1 = dout @ W2.T
    dh1 = da1 * (h1 > 0)  # ReLU grad
    grads["W1"] = X.T @ dh1 + self.reg * W1
    grads["b1"] = np.sum(dh1, axis=0)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

  def save(self, fname):
    """Save model parameters."""
    fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
    params = self.params
    np.save(fpath, params)
    print(fname, "saved.")

  def load(self, fname):
    """Load model parameters."""
    fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
    if not os.path.exists(fpath):
      print(fname, "not available.")
      return False
    else:
      params = np.load(fpath, allow_pickle=True).item()
      self.params = params
      print(fname, "loaded.")
      return True
