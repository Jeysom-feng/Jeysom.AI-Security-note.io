### <font style="color:rgb(0,0,0);">思想/流程：</font>
TRADES很大程度缓解了PGD-AT无法权衡自然准确率与鲁棒准确率的缺点，他提出了边界误差（在决策边缘化的样本被极小的扰动干扰导致分类错误）以及自然误差（在干净样本中被误分类）这两个概念；在损失函数中添加了干净样本分类结果与对抗样本分类结果的KL散度（描述干净样本分类结果与对抗样本分类结果，以及β这个超参来平衡模型的干净准确率与鲁棒准确率（模型是看重准确率还是鲁棒性）

#### 核心公式：
$ \min_f \mathbb{E}_{(x,y)\sim \mathcal{D}}
\left[
\mathcal{L}_{CE}\big(f(x),y\big)
+
\beta\cdot \max_{\|\delta\|_\infty \le \varepsilon} D_{KL}\big(f(x) \parallel f(x+\delta)\big)
\right]
 $

通过梯度下降更新可训练参数来是这个公式的值下降

###  《Theoretically Principled Trade-off between Robustness and Accuracy》  论文的实验结论
#### 一、超参数敏感性实验（Table 4）—— 直接验证 trade-off
表格

| 数据集 | (1/\lambda) | 自然准确率 | 鲁棒准确率 |
| --- | --- | --- | --- |
| MNIST | 0.1 → 5.0 | 99.41% → 99.16%（几乎不掉） | 91.09% → **95.65%** |
| CIFAR-10 | 0.1 → 5.0 | 91.31% → **81.72%**（明显掉） | 26.53% → **50.64%** |


结论：β越大，鲁棒性升、干净准确率降，与理论预测完全一致；MNIST 任务简单所以几乎不付代价。论文推荐 β in [1,10]，实测取 **1 或 6**。

#### 二、白盒攻击（Table 5，最核心结果）
**CIFAR-10（WRN-34-10）：**

表格

| 防御方法 | 自然准确率 | PGD-20 鲁棒 | PGD-1000 鲁棒 |
| --- | --- | --- | --- |
| 梯度混淆类防御（ICLR'18 那批） | — | **0%–15%**（强攻击下全崩） | — |
| Madry AT | 94.64% | 47.04% | — |
| **TRADES（β=1）** | 88.64% | 49.14% | 48.90% |
| **TRADES（β=6）** | 84.92% | **56.61%** | **56.43%** |


其他攻击下 β=6 的鲁棒准确率：DeepFool L∞ 61.38%、C&W 81.24%、MI-FGSM 57.95%。**即使把 PGD 步数从 20 加到 1000，鲁棒率几乎不衰减**，说明不是靠梯度混淆 "假鲁棒"。

**MNIST：** TRADES（β=6）自然 99.48%、PGD-40 鲁棒 96.07%、PGD-1000 鲁棒 95.60%、C&W 下高达 99.46%，与 Madry（96.01%）持平略优。

#### 三、黑盒攻击（Table 6/7）
表格

| 攻击迁移来源 | Madry | TRADES |
| --- | --- | --- |
| CIFAR-10，从自然模型迁移 | 84.39% | **87.60%** |
| CIFAR-10，从对方模型迁移 | 66.00% | **70.14%** |


TRADES 在黑盒设定下同样全面优于 Madry，且它自己产生的对抗样本迁移性更强（边界更平滑）。

#### 四、实战检验：NeurIPS 2018 对抗视觉挑战赛
+ 数据集 Tiny ImageNet（200 类、55 万图），**纯黑盒**攻防；
+ 基于 TRADES 的方案在约 **2000 支队伍中拿第一名**；
+ 以 " 使模型出错所需的最小平均L2扰动距离 " 计，**领先第二名 11.41%**。

#### 五、附加发现：鲁棒模型可解释性更好
TRADES 训出的模型，其决策边界附近的对抗图会**同时呈现两个类别的明显特征**（如鸟 / 自行车混合体）；在 bird-or-bicycle 数据集的空间变换边界攻击下仍保持 **95% 准确率**（80% 覆盖率），说明它学到的是人类可理解的语义特征，而非脆弱的像素相关性。

---

#### **总结**：
TRADES 用一个可调旋钮 β 首次让准确率 - 鲁棒性的取舍变得可控，在 CIFAR-10 上将 SOTA 白盒鲁棒率从～47% 推到 **56.6%**，黑盒和实战比赛中均胜出，且没有梯度混淆类防御 "一加攻击步数就崩溃" 的毛病



### 基于CIFAR-10数据集的对抗训练实验：
#### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ud6d4455b" class="ne-image">

#### 代码：
```python
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
```

#### 实验结果：
模型过小导致梯度爆炸，后续对模型进行修改后继续实验
