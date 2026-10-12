

FGSM/PGD是一种基于梯度快速生成对抗样本的方法

#### 原理：
对输入图片的像素矩阵的元素数值进行修改（该改变不能够由人眼辨识出）从而达到使模型分类错误；在分类卷积神经网络中，训练模型就是不断降低损失函数值的过程，想要降低损失函数的数值就要改变模型的参数；在一般的训练过程中采用梯度下降的方法进行更新参数，在损失函数对参数如w、b进行求偏导，就可以知道该参数的增大与减小对损失函数值的影响；而FGSM/PGD是通过改变图片像素值的方法来导致模型分类错误即损失函数值增大；于是，我们在损失函数对输入的特征值求偏导就有了以下公式：

$$

\begin{aligned}
\eta &= \varepsilon \mathrm{sign}\left( \nabla_x J(\theta, x, y) \right) \\
\tilde{x} &= x + \eta
\end{aligned}

$$

##### 公式理解：
当对x的偏导的值是大于0时，说明x的增大会导致损失函数的增加，而增大的数值会被ε与sign()函数控制在[-1,1]之间

ε是人为设定样本改变的步长相当于学习率

对抗样本的确定跟参数的确定过程相反，参数的更新是沿着损失函数值减小的方向，而对抗样本的生成是沿着损失损失函数值增大的方向

sign()函数的图像如图：

<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791239965588-630721fe-f550-40a4-af07-a04574e3380c.png" width="282.14284593150774" alt="" title="" crop="0,0,1,1" id="ub1848e41" class="ne-image">



#### 基于CIFAR-10数据集的对抗测试(PGD)：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ud6d4455b" class="ne-image">

##### 代码：
```python
from cProfile import label

from sklearn import datasets
from sympy import false
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


batchsize = 8
#学习率集合 对抗样本x = 源数据 + epsilon*sign(梯度)
epsilons = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]
#不同epsilon下的对抗样本
sample = []
label_list=[]

#加载CIFAR-10数据集
def creatDataset():
    train_dataset = CIFAR10(root="D:\\Desktop\\Python_file\\DL_Project\\data", train=True, download=False, transform=ToTensor())
    test_dataset = CIFAR10(root="D:\\Desktop\\Python_file\\DL_Project\\data", train=False,download=False,transform=ToTensor())
    return train_dataset, test_dataset

class imgclassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #第一个卷积层，输入通道数为3，输出通道数为6，卷积核大小为3，步长为1
        self.conv1 = nn.Conv2d(3,6,3,1)
        #第一个池化层，最大池化层，池化核大小为2，步长为2,填充为0
        self.pooling = nn.MaxPool2d(2,2,0)  
        #第二个卷积层，输入通道数为6，输出通道数为16，卷积核大小为3，步长为1
        self.conv2 = nn.Conv2d(6,16,3,1)
        #第二个池化层，最大池化层，池化核大小为2，步长为2,填充为0
        self.pooling2 = nn.MaxPool2d(2,2,0)
        #第一个全连接层，输入特征数为16*6*6，输出特征数为120
        self.fc1 = nn.Linear(16*6*6,120)
        #第二个全连接层，输入特征数为120，输出特征数为84
        self.fc2 = nn.Linear(120,84)
        #第三个全连接层，输入特征数为84，输出特征数为10
        self.output = nn.Linear(84,10)

    def forward(self,x):
        #第一层卷积+激活函数（ReLU）+池化
        x = self.pooling(torch.relu(self.conv1(x)))
        #第二层卷积+激活函数（ReLU）+池化
        x = self.pooling2(torch.relu(self.conv2(x)))
        #将多维张量展平为一维张量
        x = x.reshape(x.size(0),-1)
        #print("展平后的张量形状：", x.shape)
        #第一层全连接+激活函数（ReLU）
        x = torch.relu(self.fc1(x))
        #第二层全连接+激活函数（ReLU）
        x = torch.relu(self.fc2(x))
        #第三层全连接
        x = self.output(x)
        return x


    
def train(model,train_dataset,):
    #数据加载器(50000条)，每批8条数据，打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    #参数优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #损失函数
    criterion =nn.CrossEntropyLoss()
    #训练轮数
    epochs =50
    for epoch in range(epochs):
        total_loss,total_sample,total_correct,start = 0,0,0,time.time()
        for x,y in train_loader:
            #前向传播
            y_pred = model(x)
            #计算损失
            loss = criterion(y_pred,y)
            #清空梯度
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()
            #训练信息可视化
            #记录预测正确的样本个数
            total_correct += (torch.argmax(y_pred,dim=-1)==y).sum()
            #记录当前总损失
            total_loss+=loss.item()*len(y)
            #记录
            total_sample += len(y)
        #打印训练信息
        print(f"Epoch: {epoch+1}, acc:{total_correct/total_sample},Loss: {total_loss/batchsize}, Time: {time.time() - start}")
    torch.save(model.state_dict(), "D:\\Desktop\\Python_file\\DL_Project\\model\\CNNmodel.pth")



def demo1(model,test_dataset):
     
        #创建数据加载器
        testLoader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
        #定义变量记录正确预测的样本数和总样本数
        correct = 0
            #从数据加载器中取出每个批次的数据，进行预测
        for x,y in testLoader:
                #切换模型状态为测试模式
                model.eval()
                #神经网络最后一层没有softmax，所以输出的是加权求和的结果
                y_pred = model(x)
                #根据加权求和的结果得到类别，用argmax函数得到最大值的索引，即为预测类别
                predicted = torch.argmax(y_pred, dim=1)
                #统计正确预测的样本数
                correct += (predicted == y).sum()
        print(f"Test Accuracy: {correct / len(test_dataset) * 100:.2f}%")

#投影梯度下降法生成对抗样本
def PGD(test_dataset, epsilons, sample, model):
    model.eval()
    train_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    for ep in epsilons:                         
        steps = 10                              
        adv_batches = []                         
        y_batches = []                        
        for x, y in train_loader:              
            x_adv = x.clone().detach()         
            for step in range(steps):            
                x_adv = x_adv.detach().requires_grad_(True)
                y_pred = model(x_adv)
                loss = criterion(y_pred, y)
                model.zero_grad()
                loss.backward()
                grad = x_adv.grad.data
                x_adv = x_adv.detach() + (ep/steps) * torch.sign(grad)
                x_adv = torch.clamp(x_adv, x-ep, x+ep)
                x_adv = torch.clamp(x_adv, 0, 1)
            adv_batches.append(x_adv.detach())   
            y_batches.append(y.detach())          
        sample.append(torch.cat(adv_batches))     
        label_list.append(torch.cat(y_batches))    

 
#对抗测试
def demo2(model, sample, label_list, epsilons):
    model.eval()
    MissRate = []
    for demo, y_true in zip(sample, label_list):
        with torch.no_grad():
            pred = torch.argmax(model(demo), dim=1)
        acc = (pred == y_true).float().mean().item()
        MissRate.append(1 - acc)
    print("MissRate：", MissRate)

    
    

    #绘制折线图
    plt.plot(epsilons,MissRate,marker = "o",linestyle = "-")
    plt.xlabel("epsilon")
    plt.ylabel("Missrate")
    plt.title("Error rate under different epsilon (PGD attack)")
    plt.grid(True) # 显示网格线
    plt.show()
    # print(f"Test Accuracy: {correct / len(test_dataset) * 100:.2f}%")

#测试
if __name__ == "__main__":
    #加载模型
   if __name__ == "__main__":
    model = imgclassifier()
    model.load_state_dict(torch.load("D:\\Desktop\\Python_file\\DL_Project\\model\\CNNmodel.pth"))
    train_dataset, test_dataset = creatDataset()
    PGD(test_dataset, epsilons, sample, model)              
    demo2(model, sample, label_list, epsilons)             

```

##### 实现时遇到的问题：
1. 输入张量`x`默认无梯度，不能直接`x.grad.zero_()`，需要手动开启`requires_grad=True`。
2. 循环结构错误，多余`epoch`训练循环，PGD 不需要数据集训练轮次，只需对样本迭代扰动。
3. 缺少`model.eval()`，模型处于训练模式，BN/Dropout 会更新，干扰攻击结果。
4. 缺少**投影裁剪**`torch.clamp`，无法限制扰动范围，不是真正的 PGD。
5. 梯度未正确清零，参数梯度会累积，需要`model.zero_grad()`。
6. 将带计算图的张量存入 sample 列表，显存占用高，需要`.detach()`剥离计算图。
7. 在 PGD 函数内部新建模型实例，错误；应传入**已经训练好的外部模型**。

##### 对抗样本与标签的处理思路：
在生产对抗样本时，把每个batch的特征与标签分别append到连两个临时列表中在一个epsilon循环结束后，将该两个临时列表分别使用cat拼接成一个，例如sample.append(torch.cat(adv_batches))

在测试时就不用构建dataloader了，直接使用for demo, y_true in zip(sample, label_list):该循环算出每个epsilon值对应的错误率



#### 实验结果：
##### 未加噪音时的样本正确率：<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791525997316-06d449ea-a1a3-411c-90ed-b511eb82ed5f.png" width="1180.3571378318022" alt="" title="" crop="0,0,1,1" id="u386229b3" class="ne-image">
##### 当epsilon=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]时错误率：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791471845564-3b71f5fc-d916-4e15-b1d6-7836585d1f56.png" width="640" alt="" title="" crop="0,0,1,1" id="u50655b6a" class="ne-image">

##### 当epsilons = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]时<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791471920692-9fe718e8-8e0c-435a-a341-2f9181cdb749.png" width="640" alt="" title="" crop="0,0,1,1" id="ufc8874af" class="ne-image">
##### 当epsilons = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015]时
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791526337177-a230b1f0-9b96-4f60-a290-c09df193c39e.png" width="640" alt="" title="" crop="0,0,1,1" id="u1f057f38" class="ne-image">

##### 原样本与对抗样本：第二张图对比
epsilons = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015]

<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791529860273-3f94d887-bbd6-4068-be10-32c547f55884.png" width="611.607140253241" alt="" title="" crop="0,0,1,1" id="ucad40c61" class="ne-image">


