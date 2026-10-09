# 梯度检查
import numpy as np
from model import MLP
from losses import cross_entropy

model = MLP([10, 5, 3], seed=0)
X = np.random.randn(4, 10).astype(np.float32)
y = np.array([0, 1, 2, 1])

logits, caches = model.forward(X, training=False)
loss, dlogits = cross_entropy(logits, y)
grads = model.backward(caches, dlogits)

eps = 1e-4
k, i, j = 'w1', 0, 0
old = model.params[k][i, j]

model.params[k][i, j] = old + eps
logits1, _ = model.forward(X, training=False)
loss1, _ = cross_entropy(logits1, y)

model.params[k][i, j] = old - eps
logits2, _ = model.forward(X, training=False)
loss2, _ = cross_entropy(logits2, y)

model.params[k][i, j] = old

num_grad = (loss1 - loss2) / (2 * eps)
ana_grad = grads[k][i, j]
print("数值梯度:", num_grad)
print("解析梯度:", ana_grad)
print("相对误差:", abs(num_grad - ana_grad) / max(1e-8, abs(num_grad) + abs(ana_grad)))