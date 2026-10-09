# 实验
import numpy as np
from model import MLP
from losses import cross_entropy, mse_softmax, svm_hinge
from optimizers import SGD, Momentum, AdaGrad, RMSProp, Adam, AdaMax
from utils import run_experiment

# 载入数据
data = np.load('mnist_split.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

'''
# ==================== 学习率对比 ====================
# exp01: baseline

run_experiment(
    'exp01_baseline',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=20, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0,
    patience=None          # 不早停
)

# exp02: lr0.01
run_experiment(
    'exp02_lr0.01',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.01),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=20, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0,
    patience=None          # 不早停
)

# exp03: lr0.05
run_experiment(
    'exp03_lr0.05',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.05),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=20, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0,
    patience=None          # 不早停
)

# ==================== 网络结构对比 ====================
# exp04: 小网络
run_experiment(
    'exp04_struct_128',
    MLP([784, 128, 10], seed=0),
    Adam(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp05: 中网络
run_experiment(
    'exp05_struct_256',
    MLP([784, 256, 10], seed=0),
    Adam(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp06: 深网络
run_experiment(
    'exp06_struct_256_128',
    MLP([784, 256, 128, 10], seed=0),
    Adam(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

#发现了过拟合问题 改train.py 改用val_loss来判断最佳的epoch 同时使用早停

# ==================== 网络结构影响+早停+改用val_loss寻找best实验 ====================
# exp07: 小网络 
run_experiment(
    'exp07_earlystop_128',
    MLP([784, 128, 10], seed=0),
    Adam(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0,
    patience=None          # 不早停          
)

# exp08: 中网络 + 早停
run_experiment(
    'exp08_earlystop_256',
    MLP([784, 256, 10], seed=0),
    Adam(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0,
    patience=10   # ← 早停：连续 10 轮 val_loss 不降就停
)

# exp09: 深网络 + 早停
run_experiment(
    'exp09_earlystop_256_128',
    MLP([784, 256, 128, 10], seed=0),
    Adam(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0,
    patience=10
)'''
# MNIST 的特征很简单，网络加深反而容易过拟合，而且acc效果也不明显
# 后期可换数据集尝试?(或尝试探索正则化？)
#==========损失函数对比==========
# exp10: 交叉熵
'''run_experiment(
    'exp10_loss_ce',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp11: MSE + Softmax
run_experiment(
    'exp11_loss_mse',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=mse_softmax,
    epochs=80, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp12: SVM Hinge
run_experiment(
    'exp12_loss_svm',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=svm_hinge,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)
'''
# ==================== 正则化对比 ====================
'''
# exp13: 无正则（对照）
run_experiment(
    'exp13_reg_none',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp14: L2 1e-4
run_experiment(
    'exp14_reg_l2_1e4',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=1e-4, reg_type='L2',
    seed=0, patience=None
)

# exp15: L2 1e-3
run_experiment(
    'exp15_reg_l2_1e3',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=1e-3, reg_type='L2',
    seed=0, patience=None
)

# exp16: L1 1e-5
run_experiment(
    'exp16_reg_l1_1e5',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=80, batch_size=64,
    reg=1e-5, reg_type='L1',
    seed=0, patience=None
)

# exp17: Dropout 0.2
run_experiment(
    'exp17_reg_dropout',
    MLP([784, 128, 10], dropout=0.2, seed=0),   # ← Dropout 在模型里
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)'''
# ==================== 优化器对比 ====================
# exp18: Batch GD（全批量）
'''run_experiment(
    'exp18_batchgd',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=50000,   # ← 全批量
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp19: Online GD 单样本
run_experiment(
    'exp19_onlinegd',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.01),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=10, batch_size=1,       # ← 单样本
    reg=0.0, reg_type=None,
    seed=0, patience=None
)
'''
# exp20: Mini-batch SGD
run_experiment(
    'exp20_sgd',
    MLP([784, 128, 10], seed=0),
    SGD(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp21: Momentum
run_experiment(
    'exp21_momentum',
    MLP([784, 128, 10], seed=0),
    Momentum(lr=0.1),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp22: AdaGrad
run_experiment(
    'exp22_adagrad',
    MLP([784, 128, 10], seed=0),
    AdaGrad(lr=0.01),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp23: RMSProp
run_experiment(
    'exp23_rmsprop',
    MLP([784, 128, 10], seed=0),
    RMSProp(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp24: Adam
run_experiment(
    'exp24_adam',
    MLP([784, 128, 10], seed=0),
    Adam(lr=0.001),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)

# exp25: AdaMax
run_experiment(
    'exp25_adamax',
    MLP([784, 128, 10], seed=0),
    AdaMax(lr=0.002),
    X_train, y_train, X_val, y_val, X_test, y_test,
    loss_fn=cross_entropy,
    epochs=40, batch_size=64,
    reg=0.0, reg_type=None,
    seed=0, patience=None
)
