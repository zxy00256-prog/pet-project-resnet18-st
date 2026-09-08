import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 包装类：给Subset子集附加独立transform，不污染原始数据集
class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_dataloaders(batch_size=32):
    dataset_root = r"C:\Users\universe\Desktop\pet_project2\pet_project\data\petdata"
    print(f"[DATASET LOG] 使用数据集路径: {dataset_root}")

    full_ds = datasets.OxfordIIITPet(
        root=dataset_root,
        split="trainval",
        download=False,
        target_types="category"
    )
    print(f"[DATASET LOG] 成功加载原始数据集，总样本数: {len(full_ds)}")

    indices = list(range(len(full_ds)))
    labels = [int(full_ds[i][1]) for i in indices]

    idx_train, idx_rest, _, label_rest = train_test_split(
        indices, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_rest, label_rest, test_size=0.5, stratify=label_rest, random_state=SEED
    )

    # 先生成原始子集（不带transform）
    raw_train_sub = Subset(full_ds, idx_train)
    raw_val_sub = Subset(full_ds, idx_val)
    raw_test_sub = Subset(full_ds, idx_test)

    # 通过包装类给每个子集绑定独立transform，__getitem__内部直接返回tensor+int标签
    train_ds = TransformSubset(raw_train_sub, transform=train_transform)
    val_ds = TransformSubset(raw_val_sub, transform=val_test_transform)
    test_ds = TransformSubset(raw_test_sub, transform=val_test_transform)

    # 现在dataset输出已经是 tensor, int，直接使用默认collate_fn即可，不需要自定义collate
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[DATASET LOG] 划分完成｜训练集:{len(train_ds)} 验证集:{len(val_ds)} 测试集:{len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("===== Dataset单元测试开始 =====")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    imgs, labels = next(iter(train_loader))
    print(f"[CHECKPOINT] Batch图像shape: {imgs.shape}")
    print(f"[CHECKPOINT] Batch标签shape: {labels.shape}")
    print(f"[CHECKPOINT] 标签dtype: {labels.dtype}")
    print(f"[CHECKPOINT] 标签范围: [{labels.min().item()}, {labels.max().item()}]")
    print("===== Dataset单元测试结束，加载正常 =====")
