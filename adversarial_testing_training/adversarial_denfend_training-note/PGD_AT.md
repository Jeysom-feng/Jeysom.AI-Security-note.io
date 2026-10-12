### <font style="color:rgb(0,0,0);">思想/流程：</font>
<font style="color:rgb(0,0,0);">PGD-AT通过寻找当前可训练参数θ时，使交叉熵损失函数值最大的对抗样本( 先假设每个样本都被攻击者改成了最容易分错的样子（内层 max）  )，通过梯度下降更新可训练参数θ（也就是让可训练参数θ向对抗样本的交叉熵损失函数值下降方向走），按照该种步骤不断迭代</font>

#### <font style="color:rgb(0,0,0);">核心公式：</font>
$ \min_{\theta}\rho(\theta),\quad
\rho(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}}
\left[
\underbrace{\max_{\delta\in S}}_{\text{攻击者：找最坏扰动}}
\underbrace{L(\theta, x+\delta, y)}_{\text{分类损失（交叉熵）}}
\right]
 $

### Madry论文的实验结论：
#### 一、白盒鲁棒准确率（核心结果）
##### **MNIST**（$ \varepsilon=0.3 $，像素尺度 0–1）
| 攻击 | 配置 | 准确率 |
| --- | --- | --- |
| 自然样本 | — | 98.8% |
| FGSM | 单步 | 95.6% |
| PGD | 40 步 ×1 重启 | 93.2% |
| PGD | 100 步 ×20 重启（最强） | **89.3%** |
| CW（L2） | 40 步 | 94.0% |


##### **CIFAR-10**（$ \varepsilon=8/255 $）
| 攻击 | 配置 | 准确率 |
| --- | --- | --- |
| 自然样本 | — | 87.3% |
| FGSM | 单步 | 56.1% |
| PGD | 7 步 | 50.0% |
| PGD | 20 步（最强） | **45.8%** |
| CW（L2） | 30 步 | 46.8% |


#### 二、黑盒/迁移攻击
| 数据集 | 迁移 PGD | 迁移 FGSM/CW |
| --- | --- | --- |
| MNIST | — | 95.4–96.4% |
| CIFAR-10 | 64.2%（7 步） | 67.0% / 78.7% |


黑盒准确率明显高于白盒，且**攻击越强、网络越宽，可迁移性越差**。

#### 三、损失景观观察（PGD 作为"通用一阶攻击"的依据）
+ PGD 损失在少数几步内快速上升并**稳定 plateau**；
+ $ 10^5 $ 次随机重启得到的局部极大值，**损失值高度集中、无离群点**（尽管极大值点彼此相距很远、夹角约 $ 90^\circ $）；
+ 对抗训练网络的最大损失**显著低于**标准网络。

#### 四、网络容量实验
| 训练方式 | CIFAR 简单 ResNet | 加宽 10× |
| --- | --- | --- |
| 标准训练 / PGD 攻击 | 0.8% | 3.5% |
| PGD-AT / 自然样本 | 79.4% | 87.3% |
| **PGD-AT / PGD 攻击** | **43.7%** | **45.8%** |


关键结论：**容量越大，鞍点问题最优值越低、鲁棒性越好**；小容量模型在 PGD-AT 下会坍缩为"预测固定类"；FGSM 训练存在 label leaking，对 PGD 零防御。

#### 五、扰动强度 $ \varepsilon $ 的影响
+ $ \varepsilon \le $ 训练值时准确率持平或更高；
+ MNIST 一旦超过训练 $ \varepsilon $，鲁棒性**急剧下降**（阈值滤波器只针对固定 $ \varepsilon $）；CIFAR 衰减较平滑；
+ MNIST 上 PGD 会**高估 **$ \ell_2 $** 鲁棒性**（学到的阈值滤波器遮蔽梯度），用不依赖梯度的 Decision Boundary Attack 才能暴露其脆弱性。

#### **总结**：
PGD-AT 在 MNIST 上把白盒鲁棒准确率做到 89.3%、黑盒 95%+，在 CIFAR-10 上首次做到白盒 45.8%、黑盒 64%，验证了"一阶攻击损失集中 → PGD 是通用一阶对手"和"鲁棒性依赖足够网络容量"两大核心论断。

### 基于CIFAR-10数据集的对抗训练实验：
#### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ud6d4455b" class="ne-image">

#### 代码：
```python
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
```

#### 实验结果：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791797469233-ca30386c-dd08-4dba-a03d-d70406d0cf23.png" width="1303" alt="" title="" crop="0,0,1,1" id="ud53f9535" class="ne-image">

##### 实验结论：
从实验结果来看，PGD-AT确实能够提升模型的鲁棒准确率，但在本次实验中鲁棒准确率提升不大的原因系我自己建立的模型是小网络模型

从每轮的损失值来看：验证了Madry论文“ 一阶攻击损失值高度集中  ”的结论

从训练前后自然准确率的变化来看：验证了Madry论文中“ 小容量模型在 PGD-AT 下会坍缩为 "预测固定类"  的结论


