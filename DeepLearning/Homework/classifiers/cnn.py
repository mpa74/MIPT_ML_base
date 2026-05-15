import numpy as np

from .layer_utils import *
from .layers import *


class ThreeLayerConvNet:
  """
  трехслойная сверточная сеть с архитектурой:
  conv - relu - 2x2 max pool - linear - relu - linear - softmax

  Сеть работает с пакетами данных формой (N, C, H, W),
  состоящими из N изображений, каждое высотой H и шириной W и с C входными
  каналами.
  """

  def __init__(
    self,
    input_dim=(3, 32, 32),
    num_filters=32,
    filter_size=7,
    hidden_dim=100,
    num_classes=10,
    weight_scale=1e-3,
    reg=0.0,
    dtype=np.float32,
  ):
    """
    Инициализация.
    Входные параметры:
        - input_dim: кортеж (C, H, W), указывающий размер входных данных
        - num_filters: количество фильтров сверточного слоя
        - filter_size: ширина/высота фильтров сверточного слоя
        - hidden_dim: количество нейронов для использования в полносвязном скрытом слое
        - num_classes: количество оценок, получаемых из финального линейного слоя.
        - weight_scale: скаляр, указывающий стандартное отклонение для
        случайной инициализации
        весов.
        - reg: скаляр, указывающий силу L2-регуляризации
        - dtype: тип данных numpy для вычислений.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Инициализируйте веса и смещения для трехслойной сверточной сети
    # сети. Веса должны быть инициализированы гауссовым распределением с центром в 0,0
    # со стандартным отклонением, равным weight_scale; смещения должны быть
    # инициализированы нулем. Все веса и смещения должны храниться в
    # словаре self.params. Сохраняйте веса и смещения для сверточного
    # слоя, используя ключи 'W1' и 'b1'; используйте ключи 'W2' и 'b2' для
    # весов и смещений скрытого слоя, и ключи 'W3' и 'b3'
    # для весов и смещений выходного слоя.
    # #
    # ВАЖНО: паддинг и страйды
    # первого сверточного слоя выбраны таким образом, чтобы #
    # **ширина и высота входных данных сохранялись**. Взгляните на
    # начало функции loss() #
    ############################################################################

    F, (C, H, W) = num_filters, input_dim  # dim size
    HH = filter_size
    WW = filter_size

    self.params["W1"] = weight_scale * np.random.randn(F, C, HH, WW)
    self.params["b1"] = np.zeros(F)

    pool_height = 2
    pool_width = 2
    stride = 2
    H_after_pool = (H - pool_height) // stride + 1  # H//2
    W_after_pool = (W - pool_width) // stride + 1  # W//2

    fc_input_dim = F * H_after_pool * W_after_pool
    self.params["W2"] = weight_scale * np.random.randn(fc_input_dim, hidden_dim)
    self.params["b2"] = np.zeros(hidden_dim)

    self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b3"] = np.zeros(num_classes)
    # self.params.update({ #...
    #                     })
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params["W1"], self.params["b1"]
    W2, b2 = self.params["W2"], self.params["b2"]
    W3, b3 = self.params["W3"], self.params["b3"]

    filter_size = W1.shape[2]
    conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

    pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

    scores = None
    ############################################################################
    # TODO: Реализовать прямой проход для трехслойной сверточной сети, #
    # вычисляя оценки классов для X и сохраняя их в переменной scores #
    #
    # #
    # вы можете использовать функции, определенные в classifiesr/layers.py и #
    # classifiers/layer_utils.py. #
    ############################################################################

    out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    N, F, H_out, W_out = out.shape
    out_flat = out.reshape(N, -1)

    out2, cache2 = affine_relu_forward(out_flat, W2, b2)

    scores, cache3 = affine_forward(out2, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: обратный проход для трехслойной сверточной сети, #
    # сохраняя функцию потерь и градиенты в переменных loss и grads. Вычислить #
    # функцию потерь данных с помощью softmax и убедиться, что grads[k] содержит градиенты
    # для self.params[k]. Не забудьте добавить L2-регуляризацию! #
    # #
    # ПРИМЕЧАНИЕ:  L2-регуляризация включает множитель #
    # равный 0,5 для упрощения выражения для градиента. #
    ############################################################################

    loss, dout = softmax_loss(scores, y)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    dout2, dw3, db3 = affine_backward(dout, cache3)
    grads["W3"] = dw3 + self.reg * W3
    grads["b3"] = db3

    dout_flat, dw2, db2 = affine_relu_backward(dout2, cache2)
    grads["W2"] = dw2 + self.reg * W2
    grads["b2"] = db2

    dout_reshaped = dout_flat.reshape(N, F, H_out, W_out)

    _, dw1, db1 = conv_relu_pool_backward(dout_reshaped, cache1)
    grads["W1"] = dw1 + self.reg * W1
    grads["b1"] = db1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
