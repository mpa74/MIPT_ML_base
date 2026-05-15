from .layers import *


def affine_relu_forward(x, w, b):
  """
  слой, выполняющий аффинное преобразование с последующей функцией активации ReLU
  Входные данные:
  - x: Входные данные для аффинного слоя
  - w, b: Веса для аффинного слоя

  Возвращает кортеж:
  - out: Выходные данные функции ReLU
  - cache: Объект для передачи обратному проходу
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Обратный проход affine-relu
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
  """
  слой, выполняющий свертку с последующей активацией функции ReLU.
  Входные данные:
  - x: Входные данные для сверточного слоя
  - w, b, conv_param: Веса и параметры для сверточного слоя

  Возвращает кортеж:
  - out: Выходные данные функции ReLU
  - cache: Объект для передачи обратному проходу
  """
  a, conv_cache = conv_forward_naive(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Обратный проход для conv-relu
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_naive(da, conv_cache)
  return dx, dw, db


def conv_bn_relu_backward(dout, cache):
  conv_cache, bn_cache, relu_cache = cache
  dan = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
  dx, dw, db = conv_backward_naive(da, conv_cache)
  return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Вспомогательный слой, выполняющий свертку, функцию активации ReLU и pooling.
  Входные данные:
  - x: Входные данные для сверточного слоя
  - w, b, conv_param: Веса и параметры для сверточного слоя
  - pool_param: Параметры для pooling
  Возвращает кортеж:
  - out: Выходные данные слоя pooling
  - cache: Объект для передачи обратному проходу
  """
  a, conv_cache = conv_forward_naive(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_naive(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Обратный проход для conv-relu-pool
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_naive(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_naive(da, conv_cache)
  return dx, dw, db
