# utils/gradcam.py
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os

class GradCAM:
    def __init__(self, model: nn.Module, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册hook
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img_tensor: torch.Tensor, target_class: int):
        """
        :param img_tensor: [1,3,224,224] 已经归一化的输入张量
        :param target_class: 目标类别id
        :return: heatmap (H,W) numpy数组
        """
        self.model.eval()
        output = self.model(img_tensor)
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Grad‑CAM核心计算
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        # resize到原图大小
        _, _, H, W = img_tensor.shape
        cam = cv2.resize(cam, (W, H))
        # 归一化0~1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcam_figure(img_tensor: torch.Tensor, cam: np.ndarray, save_path: str):
    """
    将原图反归一化，叠加热力图，保存图片
    img_tensor: [1,3,224,224]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.squeeze(0).cpu().permute(1,2,0).numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32)/255.0

    overlay = 0.5 * heatmap + 0.5 * img
    overlay = np.clip(overlay,0,1)

    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


if __name__ == "__main__":
    # 模块自测
    from models.model import get_resnet18
    model = get_resnet18()
    gradcam = GradCAM(model, target_layer=model.layer4[-1])
    dummy_img = torch.randn(1,3,224,224)
    cam = gradcam(dummy_img, target_class=0)
    save_gradcam_figure(dummy_img, cam, save_path="./test_gradcam.png")
    print("✅Grad‑CAM测试图保存 test_gradcam.png")