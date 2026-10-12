import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from CNN_ImgClassfier import imgclassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 8 / 255
ALPHA = 2 / 255
TRAIN_STEPS = 10
TEST_STEPS = 20


def creatDataset():
    train_dataset = CIFAR10(root=r"D:\Desktop\Python_file\DL_Project\data", train=True, download=False, transform=ToTensor())
    test_dataset = CIFAR10(root=r"D:\Desktop\Python_file\DL_Project\data", train=False, download=False, transform=ToTensor())
    return train_dataset, test_dataset


def PGD_AT(model, train_dataset, epochs=30, batch_size=128):
    """PGD 对抗训练：内循环 PGD 生成对抗样本，外循环在对抗样本上更新参数"""
    model.to(DEVICE)
    model.train()
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    atk = torchattacks.PGD(model, eps=EPS, alpha=ALPHA, steps=TRAIN_STEPS)

    for epoch in range(epochs):
        model.train()
        total_loss, total = 0.0, 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.train()
            adv = atk(x, y)                       # 内循环：当前参数下的最坏对抗样本
            loss = criterion(model(adv), y)      # 外循环：在对抗样本上算损失
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
        print(f"Epoch {epoch + 1}/{epochs}  adv loss: {total_loss / total:.4f}")


@torch.no_grad()
def eval_clean(model, test_dataset, batch_size=256):
    """自然准确率：干净测试集"""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        predicted = model(x).argmax(dim=1)
        correct += (predicted == y).sum().item()
    return correct / len(test_dataset)


def eval_pgd(model, test_dataset, batch_size=256, steps=TEST_STEPS):
    """鲁棒准确率：测试时实时生成 PGD 对抗样本（步数比训练时更强）"""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    atk = torchattacks.PGD(model, eps=EPS, alpha=ALPHA, steps=steps)
    correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        adv = atk(x, y)                          # 攻击必须在梯度开启状态下进行，不能整体 no_grad
        with torch.no_grad():
            predicted = model(adv).argmax(dim=1)
        correct += (predicted == y).sum().item()
    return correct / len(test_dataset)


if __name__ == "__main__":
    model = imgclassifier().to(DEVICE)
    model.load_state_dict(torch.load(r"D:\Desktop\Python_file\DL_Project\model\CNNmodel.pth", weights_only=True))
    train_dataset, test_dataset = creatDataset()

    print(f"训练前 自然准确率: {eval_clean(model, test_dataset) * 100:.2f}%")
    print(f"训练前 鲁棒准确率: {eval_pgd(model, test_dataset) * 100:.2f}%")

    PGD_AT(model, train_dataset, epochs=30)
    torch.save(model.state_dict(), r"D:\Desktop\Python_file\DL_Project\model\CNNmodel_pgd_at.pth")

    print(f"训练后 自然准确率: {eval_clean(model, test_dataset) * 100:.2f}%")
    print(f"训练后 鲁棒准确率: {eval_pgd(model, test_dataset) * 100:.2f}%")
