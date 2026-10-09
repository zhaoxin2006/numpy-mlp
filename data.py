import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    # 打乱顺序
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, y_train = X[:50000], y[:50000]
    X_val, y_val = X[50000:60000], y[50000:60000]
    X_test, y_test = X[60000:], y[60000:]

    # 保存为压缩文件
    np.savez_compressed(
        'mnist_split.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print("保存完成：mnist_split.npz")
    print("训练集", X_train.shape, "验证集", X_val.shape, "测试集", X_test.shape)

if __name__ == '__main__':
    load_mnist()