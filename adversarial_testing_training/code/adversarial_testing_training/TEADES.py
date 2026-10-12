import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchattacks
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from CNN_ImgClassfier import imgclassifier
import torch.nn.functional as F
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 8 / 255
ALPHA = 2 / 255
TRAIN_STEPS = 10
TEST_STEPS = 20


def creatDataset():
    train_dataset = CIFAR10(root=r"D:\Desktop\Python_file\DL_Project\data", train=True, download=False, transform=ToTensor())
    test_dataset = CIFAR10(root=r"D:\Desktop\Python_file\DL_Project\data", train=False, download=False, transform=ToTensor())
    return train_dataset, test_dataset


def train_trades(model, train_dataset, epochs=30, batch_size=128,
                 eps=EPS, alpha=ALPHA, steps=TRAIN_STEPS, beta=1.0):
    model.to(DEVICE)
    train_dataset.transform = transforms.Compose([     # 论文标准增强
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    kl = nn.KLDivLoss(reduction='sum')
    atk = torchattacks.TPGD(model, eps=eps, alpha=alpha, steps=steps)  # 内层最大化 KL

    for epoch in range(epochs):
        model.train()
        tot_loss, tot = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = atk(x, y)                          # 内层 TPGD
            model.train()
            logits, logits_adv = model(x), model(x_adv)
            loss = F.cross_entropy(logits, y) + beta * (1.0/x.size(0)) * kl(
                F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()*x.size(0); tot += x.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} loss={tot_loss/tot:.4f}")


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

    train_trades(model, train_dataset)
    torch.save(model.state_dict(), r"D:\Desktop\Python_file\DL_Project\model\CNNmodel_trades.pth")

    print(f"训练后 自然准确率: {eval_clean(model, test_dataset) * 100:.2f}%")
    print(f"训练后 鲁棒准确率: {eval_pgd(model, test_dataset) * 100:.2f}%")
