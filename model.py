# 写全连接网络
import numpy as np

class MLP:
    def __init__(self, sizes, dropout=0.0, seed=0):
        self.sizes = sizes     # 每层神经元数，含输入和输出
        self.dropout = dropout   # 丢弃率
        self.rng = np.random.default_rng(seed)    # 固定种子的随机数生成器
        self.params = {}       # 存储权重和偏置

        #逐层初始化权重和偏置
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]   # 本层输入神经元数
            # He初始化权重，偏置初始化为0
            w = self.rng.normal(0, np.sqrt(2.0/fan_in), size = (sizes[i],sizes[i+1])).astype(np.float32)
            b = np.zeros(sizes[i+1], dtype=np.float32)
            self.params[f'w{i+1}'] = w
            self.params[f'b{i+1}'] = b 

    def forward(self, X, training=True):
        caches = {'A0' : X}  # 存储每层的输入和输出
        A = X
        L = len(self.sizes)-1 # 网络层数

        # 遍历中间隐藏层
        for i in range(1, L):
            Z = A @ self.params[f'w{i}'] + self.params[f'b{i}'] 
            A = np.maximum(0, Z) # ReLU激活函数

            if training and self.dropout>0.0:
                mask = (self.rng.random(A.shape)>self.dropout).astype(A.dtype)# 生成丢弃掩码,是一个bool数组，true表示保留，false表示丢弃,再转为float32类型
                A = A*mask # 将该丢弃的神经元输出置为0
                A = A/(1-self.dropout) # 保留的神经元输出放大，保证期望不变
                caches[f'drop{i}'] = mask # 保存丢弃掩码

            caches[f'Z{i}'] = Z
            caches[f'A{i}'] = A

        # 输出层
        Zout = A @ self.params[f'w{L}'] + self.params[f'b{L}'] #只线性变换，先不加激活函数
        caches[f'Z{L}'] = Zout
        caches[f'A{L}'] = Zout
        return Zout, caches

    def backward(self, caches, dlogits):
        grads = {} #存储每层的梯度
        L = len(self.sizes)-1 # 网络层数

        # 计算输出层的梯度
        A_prev = caches[f'A{L-1}'] #输出层的输入
        grads[f'dw{L}'] = A_prev.T @ dlogits
        grads[f'db{L}'] = dlogits.sum(axis=0)

        dA = dlogits @ self.params[f'w{L}'].T

        # 反向传播隐藏层
        for i in range(L-1, 0 ,-1):
            if self.dropout>0.0 and caches.get(f'drop{i}') is not None:
                dA = dA*caches[f'drop{i}']/(1-self.dropout) 

            dZ = dA*(caches[f'Z{i}']>0) # 括号中为ReLU的导数，Z>0时为1，否则为0
            grads[f'dw{i}'] = caches[f'A{i-1}'].T @ dZ 
            grads[f'db{i}'] = dZ.sum(axis=0)
            dA = dZ @ self.params[f'w{i}'].T

        return grads
        







        