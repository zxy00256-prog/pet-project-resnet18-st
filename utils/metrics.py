# utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    计算top‑1 accuracy、macro‑f1
    :param y_true: 真实标签 1‑D array
    :param y_pred: 预测标签 1‑D array
    :return: dict{top1_acc, macro_f1}
    """
    top1_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "top1_acc": round(float(top1_acc), 4),
        "macro_f1": round(float(macro_f1), 4)
    }


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: str = "./confusion_matrix.png",
                          figsize=(14, 14)):
    """绘制并保存混淆矩阵热力图，37类"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix (Oxford‑IIIT‑Pet 37 Classes)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


if __name__ == "__main__":
    # 简单自测
    yt = np.array([0,1,2,0,1,2])
    yp = np.array([0,2,2,0,1,1])
    res = compute_metrics(yt, yp)
    print(res)
    plot_confusion_matrix(yt, yp, save_path="./test_cm.png")
    print("混淆矩阵测试图已保存 test_cm.png")