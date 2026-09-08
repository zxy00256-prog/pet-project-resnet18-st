import numpy as np
import matplotlib.pyplot as plt

log = np.load("train_log.npy")
# 列顺序:[train_loss,train_top1,train_top5,val_loss,val_top1,val_top5]
train_loss = log[:, 0]
train_acc = log[:, 1]
val_loss = log[:, 3]
val_acc = log[:, 4]

plt.figure(figsize=(12, 5))

# 左图 Loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="train loss", color="#1f77b4")
plt.plot(val_loss, label="val loss", color="#ff7f0e")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 右图 Acc
plt.subplot(1, 2, 2)
plt.plot(train_acc, label="train acc", color="#1f77b4")
plt.plot(val_acc, label="val acc", color="#ff7f0e")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()   # 自动调整边距，防止标签被截断

# =====保存图片！！放在plt.show()前面=====
plt.savefig("loss_acc_curve.png", dpi=300, bbox_inches="tight")
print("✅图片保存为 loss_acc_curve.png")

plt.show()
