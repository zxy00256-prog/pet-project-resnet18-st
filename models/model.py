# models/model.py
import torch
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes: int = 37, pretrained: bool = True):
    """
    获取ResNet18模型，替换最后分类头适配Oxford‑IIIT‑Pet 37类
    :param num_classes: 输出类别数，宠物数据集固定37
    :param pretrained: 是否加载ImageNet预训练权重（迁移学习）
    :return: model
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # 获取fc层输入维度，替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    # 模块自测
    model = get_resnet18(num_classes=37, pretrained=True)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入shape: {dummy_input.shape}")
    print(f"输出shape: {output.shape}")
    assert output.shape[-1] == 37, "分类头输出通道必须等于37"
    print("✅模型构建通过，输出37维logits")