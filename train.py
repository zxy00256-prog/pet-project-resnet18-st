import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix

from models.model import get_resnet18
from data.dataset import get_dataloaders


# ===================== 超参数&消融开关 =====================
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4   # 从1e‑3改成1e‑4，缩小10倍！！！

# 优先cuda，失败回退cpu；遇到no‑kernel报错就直接写死cpu
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")   # 强制CPU，规避该CUDA报错

PRETRAINED = True
USE_MIXUP = False
LABEL_SMOOTH_EPS = 0.1   # 打开标签平滑


SAVE_CKPT = "./checkpoints"
LOG_DIR = "./runs/baseline"
os.makedirs(SAVE_CKPT, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

# -------------------------- 新增：训练日志容器 --------------------------
# 每一行：[train_loss, train_top1, train_top5, val_loss, val_top1, val_top5]
train_log = []

# ===================== 加载数据集 =====================
train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# -------- Day1检查点：打印Dataloader batch shape --------
imgs_sample, label_sample = next(iter(train_loader))
print(f"[Dataloader Check] batch shape: {imgs_sample.shape}")

# ===================== 模型、损失、优化器 =====================
model = get_resnet18(num_classes=37, pretrained=PRETRAINED).to(DEVICE)

if LABEL_SMOOTH_EPS > 0:
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH_EPS)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

# ===================== 指标函数 =====================
def calc_top1_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean()

def calc_top5_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, top5_idx = torch.topk(logits, k=5, dim=-1)
    labels_exp = labels.view(-1, 1)
    correct = torch.any(top5_idx == labels_exp, dim=1)
    return correct.float().mean()

# ===================== MixUp工具 =====================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_loss(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

best_val_top1 = 0.0
best_ckpt_path = os.path.join(SAVE_CKPT, "best_resnet18.pth")

# ===================== 主训练循环 =====================
for epoch in range(1, EPOCHS + 1):
    # -------- Train --------
    model.train()
    tr_loss_sum = 0.0
    tr_top1_sum = 0.0
    tr_top5_sum = 0.0
    tr_total = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        bsz = imgs.shape[0]
        optimizer.zero_grad()

        if USE_MIXUP:
            imgs_mix, y_a, y_b, lam = mixup_data(imgs, labels)
            logits = model(imgs_mix)
            loss = mixup_loss(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if USE_MIXUP:
                logits_raw = model(imgs)
                top1 = calc_top1_acc(logits_raw, labels)
                top5 = calc_top5_acc(logits_raw, labels)
            else:
                top1 = calc_top1_acc(logits, labels)
                top5 = calc_top5_acc(logits, labels)

        tr_loss_sum += loss.item() * bsz
        tr_top1_sum += top1.item() * bsz
        tr_top5_sum += top5.item() * bsz
        tr_total += bsz

    train_loss = tr_loss_sum / tr_total
    train_top1 = tr_top1_sum / tr_total
    train_top5 = tr_top5_sum / tr_total

    # -------- Validation --------
    model.eval()
    val_loss_sum = 0.0
    val_top1_sum = 0.0
    val_top5_sum = 0.0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            bsz = imgs.shape[0]
            logits = model(imgs)
            loss = criterion(logits, labels)

            top1 = calc_top1_acc(logits, labels)
            top5 = calc_top5_acc(logits, labels)

            val_loss_sum += loss.item() * bsz
            val_top1_sum += top1.item() * bsz
            val_top5_sum += top5.item() * bsz
            val_total += bsz

    val_loss = val_loss_sum / val_total
    val_top1 = val_top1_sum / val_total
    val_top5 = val_top5_sum / val_total

    # -------------------------- 新增：保存每轮指标到列表 --------------------------
    train_log.append([train_loss, train_top1, train_top5, val_loss, val_top1, val_top5])

    print(f"Epoch [{epoch:2d}/{EPOCHS}] "
          f"Train: loss={train_loss:.4f} top1={train_top1:.4f} top5={train_top5:.4f} | "
          f"Val: loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f}")

    # TensorBoard记录
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Acc/top1_train", train_top1, epoch)
    writer.add_scalar("Acc/top1_val", val_top1, epoch)
    writer.add_scalar("Acc/top5_train", train_top5, epoch)
    writer.add_scalar("Acc/top5_val", val_top5, epoch)

    # 保存最优模型
    if val_top1 > best_val_top1:
        best_val_top1 = val_top1
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"✅更新best模型，best_val_top1={best_val_top1:.4f}")

writer.close()

# -------------------------- 新增：全部epoch结束，输出train_log.npy --------------------------
train_log_arr = np.array(train_log)
np.save("train_log.npy", train_log_arr)
print("\n📝训练日志已保存至 train_log.npy")
print(f"日志shape: {train_log_arr.shape}, 列顺序:[train_loss,train_top1,train_top5,val_loss,val_top1,val_top5]")

# ===================== Test集评估（加载best权重） =====================
print("\n" + "="*30 + " Test Set Evaluation " + "="*30)
model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
model.eval()

all_preds = []
all_gts = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_gts.extend(labels.numpy())

# Test‑Top1
test_top1 = np.mean(np.array(all_preds) == np.array(all_gts))

# Test‑Top5
test_top5_cnt = 0
test_total_cnt = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        _, top5_idx = torch.topk(logits, k=5, dim=-1)
        lab_exp = labels.view(-1,1).to(DEVICE)
        ok = torch.any(top5_idx == lab_exp, dim=1)
        test_top5_cnt += ok.sum().item()
        test_total_cnt += imgs.shape[0]
test_top5 = test_top5_cnt / test_total_cnt

# Macro‑F1
test_macro_f1 = f1_score(all_gts, all_preds, average="macro")

# 保存混淆矩阵npy
cm = confusion_matrix(all_gts, all_preds)
np.save("confusion_matrix.npy", cm)

print(f"Test Top‑1 Acc: {test_top1:.4f}")
print(f"Test Top‑5 Acc: {test_top5:.4f}")
print(f"Test Macro‑F1: {test_macro_f1:.4f}")
print("混淆矩阵已保存至 confusion_matrix.npy")
