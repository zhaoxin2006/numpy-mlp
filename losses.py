# 写损失函数
import numpy as np

def softmax(z):
    # z: (N, C) N个样本，C个类别
    # 为了数值稳定性，减去每行的最大值,exp 参数最大变成 0, 不会溢出
    # softmax(z) = softmax(z - c)（c 是常数，结果不变）
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(axis=1, keepdims=True)

# 交叉熵损失函数
def cross_entropy(logits, y):
    # logits: (N, C) N个样本，C个类别
    # y: (N,) N个样本的标签，取值范围[0, C-1]
    N = len(y)
    p = softmax(logits)
    loss = -np.log(p[np.arange(N), y]).mean() # 取出每个样本的正确类别的概率，取对数，求平均
    dlogits = p.copy()
    dlogits[np.arange(N), y] -= 1 # 对正确类别的梯度减1
    dlogits /= N
    return loss, dlogits

# mse 均方误差损失函数
def mse_softmax(logits, y):
    N = len(y)
    p = softmax(logits)
    Y = np.zeros_like(p)
    Y[np.arange(N), y] = 1
    loss = 0.5*((np.sum((p-Y)**2, axis=1)).mean())
    dp = (p-Y)/N
    dlogits = p*(dp-np.sum(dp*p, axis=1, keepdims=True))
    return loss, dlogits

# svm hinge loss 支持向量机损失函数
def svm_hinge(logits, y):
    N = len(y)
    correct = logits[np.arange(N), y][:, None] # 增加维度到(N, 1) 正确类别的logits
    margins = np.maximum(0, logits - correct+1) # 计算每个类别的margin
    margins[np.arange(N), y] = 0 # 正确类别的margin置为0
    loss = margins.sum() / N

    dlogits = np.zeros_like(logits)
    dlogits[margins>0] = 1 # margin>0的类别梯度为1,<0的类别梯度为0， margins > 0 就是一个布尔掩码
    dlogits[np.arange(N), y] = -np.sum(dlogits, axis=1) # 正确类别的梯度为负的违反类别的个数
    dlogits /= N
    return loss, dlogits




    
