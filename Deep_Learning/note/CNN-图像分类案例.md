### 代码实现：
```python
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
def train(train_dataset,):
    model =  imgclassifier()
    #数据加载器(50000条)，每批8条数据，打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    #参数优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #损失函数
    criterion =nn.CrossEntropyLoss()
    #训练轮数
    epochs =10
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
        #print(f"Epoch: {epoch+1}, acc:{total_correct/total_sample},Loss: {total_loss/batchsize}, Time: {time.time() - start}")
    torch.save(model.state_dict(), "D:\\Desktop\\Python_file\\DL_Project\\model\\CNNmodel.pth")
def demo1(test_dataset):
     #加载模型
        model = imgclassifier()
        model.load_state_dict(torch.load("D:\\Desktop\\Python_file\\DL_Project\\model\\CNNmodel.pth"))
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
    
    
#测试
if __name__ == "__main__":
    train_dataset, test_dataset =creatDataset()
    # print("训练集样本数：", len(train_dataset), "测试集样本数：", len(test_dataset))
    # print("训练集样本：", train_dataset[0][0].shape, "标签：", train_dataset[0][1])
    train(train_dataset)
    demo1(test_dataset)
```

### 优化思路：
1.增加卷积核输出通道数

2.增加全连接层的参数量

3.调整学习率

4.调整优化方法

5.修改激活函数




