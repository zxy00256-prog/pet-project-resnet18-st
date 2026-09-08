import torch
from data.dataset import get_dataloaders
from models.model import get_resnet18

def evaluate(model_path):
    device = torch.device("cpu")
    _, test_loader = get_dataloaders()
    model = get_resnet18(num_classes=37).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    evaluate("./checkpoint/best.pth")
