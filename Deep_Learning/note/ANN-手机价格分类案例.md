### 流程：
#### 数据集构建：
1.读取csv文件

2.将特征列与标签列分离

3.将特征列做类型转换，转换为float32

4.分离测试集与训练集

5.将测试集与训练集转换为张量再转化为数据集，方便后面转换为数据加载器

#### 模型构建：
1.定义一个类继承nn.module

2.重写__init__方法，先初始化super，super().__init__

3.搭建隐藏层与输出层

4.重写前向传播算法，隐藏层（加权求和与激活函数），输出层（加权求和，若用多分类交叉熵损失函数则不用写激活函数）

#### 训练模型：
1.创建训练集数据加载器对象，用于加载数据以及每批次训练条数

2.定义损失函数（nn.CrossEntropyLoss()），用于计算损失

3.定义参数优化器，用于计算梯度以及反向传播更新参数

4.定义第一层循环，每次循环为1轮训练，第二层循环的每次循环为1个批次训练，参数的更新是以批次为单位，一个批次更新一次参数

#### 测试模型：
1.加载模型参数

2.创建测试集数据加载器对象，用于加载数据以及定义每批次训练条数

3.将数据交给模型进行预测，计算正确率



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
from torchsummary import summary                #模型结构可视化

#加载数据
def load_data():
    #加载数据
    data =pd.read_csv('D:\\Desktop\\Python_file\\DL_Project\\train.csv')
    # print(data.head())
    # print(data.shape)

    #分离特征与标签
    x,y = data.iloc[:,:-1],data.iloc[:,-1]
    #print(x.shape,y.shape)

    #将特征列的类型转换为float32,方便自动微分
    x = x.astype(np.float32)

    #划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    #数据->张量->数据集->数据加载器
    TrainDataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    TestDataset =TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))
    #--------------------------------输入特征数--------输出类别数
    return TrainDataset, TestDataset,x_train.shape[1],len(np.unique(y))

    #搭建神经网络
class phonePriceModel(nn.Module):
        #初始化神经网络，该函数类似java中的构造函数
    def __init__(self,input_size,output_size):
            super().__init__()
            #隐藏层1
            self.linear1=nn.Linear(20,128)
            #隐藏层2
            self.linear2=nn.Linear(128,256)
            #输出层
            self.output=nn.Linear(256,output_size)

    #定义前向传播算法
    def forward(self,x):
       #第一层加权求和+激活函数（Relu）
       x = torch.relu(self.linear1(x))
       #第二层加权求和+激活函数（Relu）
       x = torch.relu(self.linear2(x))
       #正常写法，但是后续会用多分类交叉熵损失函数CrossEntropyLoss()，所以不需要softmax，只做加权求和
       #x =torch.softmax(self.output(x),dim=1)
       x=self.output(x)
       return x

    #训练模型
def train(train_dataset, input_size, output_size):
     #数据加载器（1600条），每批16条数据，打乱顺序
    trainLoader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    #定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #定义变量epoch
    epochs = 100
    #定义变量，记录每次训练的损失值，训练批次数
    for epoch in range(epochs):
     total_loss =0.0
     batch_num = 0
     start_time = time.time()
     #开始各个批次训练
     for x,y in trainLoader:
         #切换模型状态,model.train()训练模式，model.eval()测试模式
         model.train()
         y_pred = model(x)
         loss = criterion(y_pred, y)
         #方向传播，优化参数
         optimizer.zero_grad()
         loss.sum().backward()
         optimizer.step()
         #累计本轮损失值
         total_loss += loss.item()
         batch_num+=1
     #本轮训练结束，打印训练信息
     end_time = time.time()
     print(f"Epoch: {epoch+1}, Loss: {total_loss/batch_num}, Time: {end_time - start_time}")
     #保存模型参数，参1：模型参数，参2：保存路径
     torch.save(model.state_dict(), "D:\\Desktop\\Python_file\\DL_Project\\model\\model.pth")
#测试模型
def evaluate(test_dataset, input_size, output_size):
    #加载模型
    model = phonePriceModel(input_size, output_size)
    model.load_state_dict(torch.load("D:\\Desktop\\Python_file\\DL_Project\\model\\model.pth"))
    #创建数据加载器
    testLoader = DataLoader(test_dataset, batch_size=8, shuffle=False)
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
if __name__ == "__main__":
    train_dataset, test_dataset, input_size, output_size = load_data()
    model = phonePriceModel(input_size, output_size)
    #计算模型参数量
    #参数1：模型，参数2：输入数据的形状（批次大小，输入特征数），每批16条，每条20列特征
    # summary(model, input_size=(16,input_size))
    #train(train_dataset, input_size, output_size)
    evaluate(test_dataset, input_size, output_size)
```

### 优化思路：
1.优化方法从SGD->Adam

2.改变学习率

3.对数据进行标准化

4.增加网络深度

5.调整训练轮数


