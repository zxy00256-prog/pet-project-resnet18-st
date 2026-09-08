# Oxford‑IIIT‑Pet 宠物分类项目
基于ResNet18实现37类宠物图像分类任务，引入标签平滑损失开展消融实验，结合Grad-CAM完成模型可解释性分析。

## 实验指标
- 数据集：Oxford‑IIIT‑Pet，共37类猫狗宠物，分层抽样划分训练/验证/测试集（7:1.5:1.5）
- Label Smooth（ε=0.1）：验证集最佳准确率**93.30%**，测试集Top-1准确率**91.30%**
> 注：验证集与测试集准确率存在一定差距，模型存在轻微过拟合；标签平滑相比普通交叉熵在测试集上小幅提升Top-1准确率与Macro-F1，获得有限泛化增益。


## 数据集准备
使用 Oxford‑IIIT‑Pet 数据集，共37类宠物。
将数据集解压放到如下路径：`./data/petdata/oxford‑iiit‑pet/`

> ⚠重要：解压完成后，`./data/petdata/oxford‑iiit‑pet/` 目录下直接看到 `images`、`annotations` 两个文件夹，不要多层嵌套。
> 错误示例：`data/petdata/oxford‑iiit‑pet/oxford‑iiit‑pet/images`（多嵌套一层会报找不到数据集）

💡提示：Windows本地调试若报数据集找不到，可临时改为本机绝对路径；提交代码必须使用相对路径，保证Linux/Colab环境可复现。

```plaintext
./data/petdata/oxford‑iiit‑pet/
├─ annotations
└─ images
```

## 项目文件结构

```
pet_project/
├── data/
│   └── dataset.py        # 数据加载、划分与 Transform（分层抽样）
├── models/
│   └── model.py          # ResNet18模型定义与Head替换
├── utils/
│   ├── metrics.py        # 准确率、F1、混淆矩阵绘制
│   ├── gradcam.py        # Grad-CAM 可视化工具
│   └── draw_curve.py     # 绘制loss&accuracy训练曲线
├── results/              # 输出产物：图片、npy日志（运行代码自动生成）
│   ├── confusion_matrix.png
│   ├── confusion_matrix.npy
│   ├── gradcam_0.png
│   ├── gradcam_1.png
│   ├── loss_acc_curve.png
│   └── train_log.npy
├── train.py              # 训练主入口，支持标签平滑损失
├── evaluate.py           # 独立评估脚本，输出指标、混淆矩阵
├── requirements.txt
└── README.md
```

## 运行完整流程

```
# 安装依赖
pip install -r requirements.txt

# 模型训练
python train.py

# 模型评估
python evaluate.py

# 绘制训练曲线
python utils/draw_curve.py
```

### 实验结果

混淆矩阵：

训练 Loss&Acc 曲线：

Grad‑CAM 热力图可视化：

**样本 1：预测正确 (Correct | GT=0)**

**样本 2：预测错误 (Wrong | GT=0, Pred=32)**