#写优化器（如何用梯度更新权重？）

import numpy as np

# SGD 随机梯度下降优化器
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def get_config(self):
        return {'lr': self.lr}
    def update(self, params, grads):
        for k in params.keys():
            params[k] -= self.lr * grads[f'd{k}']

# 动量优化器
class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None # 初始化动量为None,累积动量
    def get_config(self):
        return {'lr': self.lr, 'beta': self.beta}
    def update(self, params, grads):
        # 初始化动量
        if self.v is None:
            self.v = {}
            for k, v in params.items():
                self.v[k] = np.zeros_like(v)
        # 更新动量和参数
        for k in params.keys():
            self.v[k] = self.beta * self.v[k] + (1-self.beta)*grads[f'd{k}']
            params[k] -= self.lr * self.v[k]

# AdaGrad 自适应梯度优化器
class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.cache = None # 初始化缓存为None,累积梯度平方和
    def get_config(self):
        return {'lr': self.lr, 'eps': self.eps}
    def update(self, params, grads):
        # 初始化缓存
        if self.cache is None:
            self.cache = {}
            for k, v in params.items():
                self.cache[k] = np.zeros_like(v)
        # 更新缓存和参数
        for k in params.keys():
            self.cache[k] += grads[f'd{k}']**2
            params[k] -= self.lr*grads[f'd{k}']/(np.sqrt(self.cache[k])+self.eps)

# RMSProp 自适应学习率优化器(AdaGrad的改进版)，解决了 AdaGrad 学习率归零的问题
class RMSProp:
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = None # 初始化缓存为None,用滑动平均代替累积平方和
    def get_config(self):
        return {'lr': self.lr, 'beta': self.beta, 'eps': self.eps}
    def update(self, params, grads):
        # 初始化缓存
        if self.cache is None:
            self.cache = {}
            for k, v in params.items():
                self.cache[k] = np.zeros_like(v)
        # 更新缓存和参数
        for k in params.keys():
            self.cache[k] = self.beta*self.cache[k] + (1-self.beta)*grads[f'd{k}']**2
            params[k] -= self.lr*grads[f'd{k}']/(np.sqrt(self.cache[k])+self.eps)

# Adam 优化器（结合了动量和RMSProp的优点）
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None # 初始化一阶矩为None
        self.v = None # 初始化二阶矩为None
        self.t = 0 # 初始化时间步为0
    def get_config(self):
        return {'lr': self.lr, 'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps}
    def update(self, params, grads):
        # 初始化
        if self.m is None:
            self.m = {}
            self.v = {}
            for k, v in params.items():
                self.m[k] = np.zeros_like(v)
                self.v[k] = np.zeros_like(v)
        self.t += 1 # 时间步加1
        # 更新一阶矩和二阶矩
        for k in params.keys():
            self.m[k] = self.beta1*self.m[k] + (1-self.beta1)*grads[f'd{k}']
            self.v[k] = self.beta2*self.v[k] + (1-self.beta2)*grads[f'd{k}']**2
            # 偏差修正
            m_hat = self.m[k]/(1-self.beta1**self.t)
            v_hat = self.v[k]/(1-self.beta2**self.t)
            # 更新参数
            params[k] -= self.lr*m_hat/(np.sqrt(v_hat)+self.eps)

# AdaMax 优化器（Adam的改进版）
class AdaMax:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None # 初始化一阶矩为None
        self.u = None # 初始化二阶矩为None
        self.t = 0 # 初始化时间步为0
    def get_config(self):
        return {'lr': self.lr, 'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps}
    def update(self, params, grads):
        # 初始化
        if self.m is None:
            self.m = {}
            self.u = {}
            for k, v in params.items():
                self.m[k] = np.zeros_like(v)
                self.u[k] = np.zeros_like(v)
        self.t += 1 # 时间步加1
        # 更新一阶矩和二阶矩
        for k in params.keys():
            self.m[k] = self.beta1*self.m[k] + (1-self.beta1)*grads[f'd{k}']
            self.u[k] = np.maximum(self.beta2*self.u[k], np.abs(grads[f'd{k}']))
            # 偏差修正
            m_hat = self.m[k]/(1-self.beta1**self.t)
            # 更新参数
            params[k] -= self.lr * m_hat/(self.u[k]+self.eps)



