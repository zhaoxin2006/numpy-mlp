# 写通用运行函数（保存模型，保存训练信息，写配置文件，画图）
import os
import numpy as np
import matplotlib.pyplot as plt
from train import train, evaluate
# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_experiment(exp_name, model, optimizer,
                   X_train, y_train, X_val, y_val,
                   X_test, y_test,
                   loss_fn, epochs=20, batch_size=64,
                   reg=0.0, reg_type=None, seed=0,
                   patience=None,
                   save_dir='results'):
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 训练
    history, best_val_acc, best_val_loss, best_epoch = train(
        model, X_train, y_train, X_val, y_val,
        loss_fn=loss_fn, optimizer=optimizer,
        epochs=epochs, batch_size=batch_size,
        reg=reg, reg_type=reg_type,
        seed=seed, patience=patience, verbose=False
    )


    # 用测试集测试
    test_loss, test_acc = evaluate(model, X_test, y_test, loss_fn, reg, reg_type)

    print(f"[{exp_name}] best_val_acc = {best_val_acc:.4f} (epoch {best_epoch}) | "
          f"best_val_loss = {best_val_loss:.4f} | test_acc = {test_acc:.4f}")

    # 保存模型（最佳）
    np.savez_compressed(os.path.join(exp_dir, 'model.npz'), **model.params)

    # 保存历史
    np.save(os.path.join(exp_dir, 'history.npy'), history)

    # 保存配置
    with open(os.path.join(exp_dir, 'config.txt'), 'w') as f:
        f.write(f"experiment: {exp_name}\n")
        f.write(f"structure: {model.sizes}\n")
        f.write(f"dropout: {model.dropout}\n")
        f.write(f"optimizer: {type(optimizer).__name__}\n")
        if hasattr(optimizer, 'get_config'):
            f.write(f"optimizer_config: {optimizer.get_config()}\n")
        else:
            f.write(f"optimizer_lr: {getattr(optimizer, 'lr', 'N/A')}\n")
        f.write(f"loss_fn: {loss_fn.__name__}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"reg: {reg}, reg_type: {reg_type}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"best_val_acc: {best_val_acc:.4f}\n")
        f.write(f"best_val_loss: {best_val_loss:.4f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"test_loss: {test_loss:.4f}\n")
        f.write(f"test_acc: {test_acc:.4f}\n")

    # 画曲线
    epochs_list = [h['epoch'] for h in history]
    val_accs = [h['val_acc'] for h in history]
    

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_list, [h['train_loss'] for h in history], label='train', marker='o', markersize=3)
    axes[0].plot(epochs_list, [h['val_loss'] for h in history], label='val', marker='o', markersize=3)
    axes[0].axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'best epoch={best_epoch}')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(epochs_list, [h['train_acc'] for h in history], label='train', marker='o', markersize=3)
    axes[1].plot(epochs_list, [h['val_acc'] for h in history], label='val', marker='o', markersize=3)
    axes[1].axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'best epoch={best_epoch}')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].grid(True, alpha=0.3); axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'curves.png'), dpi=150)
    plt.close()

    return best_val_acc, test_acc