import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import DeepFool




batchsize = 8
#学习率集合 对抗样本x = 源数据 + epsilon*sign(梯度)
epsilons = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015]
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


    
def train(model,train_dataset,batchsize):
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
def demo2(model, sample, label_list, epsilons,test_dataset):
    i=0
    model.eval()
    MissRate = []
    plt.figure(figsize=(16,8))
     # 左：干净原图
    #获取原样本的第二张图片
    img_clean, label_clean = test_dataset[1]
    plt.subplot(1,8,1)
    plt.imshow(img_clean.permute(1,2,0).cpu().numpy())
    plt.title(f"origin")
    plt.axis('off')
    for demo, y_true in zip(sample, label_list):
        with torch.no_grad():
            pred = torch.argmax(model(demo), dim=1)
        acc = (pred == y_true).float().mean().item()
        MissRate.append(1 - acc)


        #复制一份用于展示图像扰动前后对比
        demo_show = demo.clone().detach()
        demo_show=demo_show.permute(0,2,3,1)
        #转换为numpy对象
        demo_show=demo_show.numpy()
        #每组对抗样本展示第一张图片
        plt.subplot(1,8,2+i)
        plt.imshow(demo_show[1])
        plt.title("demo")
        plt.axis('off')
        i+=1
        #清空画布
    plt.show()
        # print(demo.shape)
    # print("MissRate：", MissRate)

    #绘制折线图
    plt.plot(epsilons,MissRate,marker = "o",linestyle = "-")
    plt.xlabel("epsilon")
    plt.ylabel("Missrate")
    plt.title("Error rate under different epsilon (PGD attack)")
    plt.grid(True) # 显示网格线
    plt.show()
    # print(f"Test Accuracy: {correct / len(test_dataset) * 100:.2f}%")

def ART_CW(model,test_dataset):
    x = test_dataset.data.astype(np.float32) / 255.0   # ← 除以255，归一化到[0,1]
    x = x.transpose(0, 3, 1, 2)                        # NCHW
    y = np.array(test_dataset.targets)                  # ← 转numpy

    # print(x.shape)
    # print(y)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    #将model封装成ART对象
    classifier = PyTorchClassifier(
    model=model,
    loss=loss,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(0, 255)
    )

    #干净样本的正确率
    acc_clean = np.mean(np.argmax(classifier.predict(x), 1) == y)
    print(f"Clean accuracy: {acc_clean:.2%}")

    #用ART对象构造C&W攻击
    attack = CarliniL2Method(
    classifier=classifier,      # 已经包好的 PyTorchClassifier
    confidence=0.0,            # 就是 κ，0=刚成功就行；调大=要更自信的误判
    targeted=False,             # False=无定向（让它随便错）；True=定向（要骗成指定类）
    learning_rate=0.01,         # Adam 学习率
    binary_search_steps=5,     # 外层 c 二分次数（论文默认5）
    max_iter=100,             # 内层 w 最大迭代步数
    initial_const=0.01,         # c 的初始值
    batch_size=128,
    )
    #生成对抗样本
    x_adv =attack.generate(x=x)
    acc_adv = np.mean(np.argmax(classifier.predict(x_adv), 1) == y)
    print(f"adv accuracy: {acc_adv:.2%}")

def ART_DeepFool(model,test_dataset):
    x = test_dataset.data.astype(np.float32) / 255.0   # ← 除以255，归一化到[0,1]
    x = x.transpose(0, 3, 1, 2)                        # NCHW
    y = np.array(test_dataset.targets)                  # ← 转numpy
    
        # print(x.shape)
        # print(y)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    
        #将model封装成ART对象
    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0, 255)
        )
    #用ART对象构造DeepFool攻击
    attack = DeepFool(
    classifier=classifier
    
    )
     #干净样本的正确率
    acc_clean = np.mean(np.argmax(classifier.predict(x), 1) == y)
    print(f"Clean accuracy: {acc_clean:.2%}")
    #生成对抗样本
    x_adv =attack.generate(x=x)
    acc_adv = np.mean(np.argmax(classifier.predict(x_adv), 1) == y)
    print(f"adv accuracy: {acc_adv:.2%}")
    pass

if __name__ == "__main__":
#测试
  #加载模型
    model = imgclassifier()
    model.load_state_dict(torch.load("D:\\Desktop\\Python_file\\DL_Project\\model\\CNNmodel.pth"))
    train_dataset, test_dataset = creatDataset()
    train(model,train_dataset,batchsize)
    # PGD(test_dataset, epsilons, sample, model)              
    # demo2(model, sample, label_list, epsilons,test_dataset)             
    demo1(model,test_dataset)
    # ART_CW(model,test_dataset)
    # ART_DeepFool(model,test_dataset)