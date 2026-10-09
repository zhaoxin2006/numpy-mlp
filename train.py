#写训练与评估
import numpy as np
from model import MLP
from losses import cross_entropy

# 正则化损失
def reg_loss(params, reg=0.0, reg_type=None):
    if reg<=0:
        return 0.0
    s = 0.0
    for k in params.keys():
        if k.startswith('w'):
            if reg_type == 'L2':
                s += 0.5*np.sum(params[k]**2)
            elif reg_type == 'L1':
                s +=np.sum(np.abs(params[k]))
    return reg*s

# 为梯度添加正则化项
def add_reg_grad(grads, params, reg=0.0, reg_type=None):
    if reg<=0:
        return 0.0
    for k in params.keys():
        if k.startswith('w'):
            if reg_type == 'L2':
                grads[f'd{k}'] = grads[f'd{k}'] + reg*params[k]
            elif reg_type == 'L1':
                grads[f'd{k}'] = grads[f'd{k}'] + reg*np.sign(params[k])

# 评估
def evaluate(model, X, y, loss_fn, reg=0.0, reg_type=None):
    logits, _ = model.forward(X, training=False)
    loss, _ = loss_fn(logits, y)
    loss += reg_loss(model.params, reg, reg_type)
    pred = logits.argmax(axis=1)
    acc = (pred==y).mean()
    return loss, acc

# 训练
def train(model, X_train, y_train, X_val, y_val, 
          loss_fn=cross_entropy, optimizer=None, 
          epochs=20, batch_size=64, reg=0.0, reg_type=None, seed=0, 
          patience=None, verbose=True):
    rng = np.random.default_rng(seed) # 随机数生成器
    history = [] # 记录每个epoch的训练和验证损失、准确率
    N = len(X_train)
    '''
    # 最佳模型记录
    best_val_acc = -1.0
    best_epoch = 0
    best_params = None
    '''
    # ===== 改动 1：用 val_loss 选 best，初始化为正无穷 =====
    best_val_loss = float('inf')
    best_val_acc = -1.0       # 顺便记录 best 对应的 val_acc
    best_epoch = 0
    best_params = None

    # 早停计数
    wait = 0
    
    for epoch in range(epochs):
        idx = rng.permutation(N) # 打乱训练集索引,生成 0..N-1 的随机排列
        for start in range(0, N, batch_size):
            bidx = idx[start:start+batch_size] # 取出一个batch的索引
            Xb, yb = X_train[bidx], y_train[bidx] # 取出一个batch的训练数据和标签

            logits, caches = model.forward(Xb, training=True) # 前向传播
            loss, dlogits = loss_fn(logits, yb) # 计算损失和梯度
            grads = model.backward(caches, dlogits) # 反向传播计算梯度
            add_reg_grad(grads, model.params, reg, reg_type) # 添加正则化梯度
            optimizer.update(model.params, grads) # 优化器更新参数

        # 每个epoch结束后评估训练集和验证集
        train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn, reg, reg_type)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn, reg, reg_type)
        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # 打印训练信息
        if verbose:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
                  f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f}"
                  )

        '''
        # 早停逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_params = {k: v.copy() for k, v in model.params.items()}
            wait = 0
        else:
            wait += 1
        '''
        # ===== 改动 2：按 val_loss 最小选 best =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_params = {k: v.copy() for k, v in model.params.items()}
            wait = 0
        else:
            wait += 1
        # ---- 早停（只在 patience 不为 None 时启用） ----
        if patience is not None and wait >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # ---- 恢复最佳模型 ----
    if best_params is not None:
        for k in model.params:
            model.params[k] = best_params[k]
 
    #return history, best_val_acc, best_epoch
    # ===== 改动 3：返回值加上 best_val_loss =====
    return history, best_val_acc, best_val_loss, best_epoch



 