### 基本构成：
输入层：输入样本特征的那一层，将数据传递给下一层

输出层：

隐藏层：输入层与输出层中间的都是隐藏层，隐藏层的神经元通过加权和激活函数处理输入，将结果传递到下一层

#### 特点：
1.同一层的神经元之间相互隔离

2.第N层神经元与第N-1层神经元全连接

3.全连接神经网络接收的样本数据是二维的，数据在每一层之间需要二维张量形式传输

4.每个连接都有一个权重值

#### 激活函数：
##### 1.sigmoid函数：
主要运用于二分类输出层，以及浅层神经网络

在数据在[-6,6]时有效果，[-3,3]时效果明显

梯度范围为[0.0.25]

<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1789981401720-834adf98-bd08-4d40-802a-decb06f6bffb.png" width="1525.892850646407" alt="" title="" crop="0,0,1,1" id="uf4046c99" class="ne-image">

缺点：在五层sigmoid激活函数神经网络后，参数几乎不更新。出现梯度消失



##### 2.Tanh函数：
主要运用于浅层神经网络（不超过五层）、隐藏层

比sigmoid函数效率高

在数据在[-3,3]时有效果，[-1,1]时效果明显

梯度范围为[0,1]

<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1789987999270-3404e334-56e4-4ccc-b642-9b4a080e2f0d.png" width="1121.4285666541177" alt="" title="" crop="0,0,1,1" id="u6fbfb2a3" class="ne-image">

##### 3.Relu函数
输入值为正时，激活函数值就是输入值

当输入值为负时，会出现神经元死亡，参数得不到更新

相较于Tanh函数收敛较快

<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1789988732094-d4ee816b-a13c-4d2d-9ff9-027561cec50a.png" width="1769.6428496086476" alt="" title="" crop="0,0,1,1" id="u3afcebb3" class="ne-image">



##### 4.SoftMax函数
将多分类的结果按概率形势展示，且相加和为1，选取概率值最大的分类做最终结果

### 参数初始化：
#### 1. 全零初始化（Zero Initialization）
+ **做法**：W = 0, b = 0（或小常数）
+ **适用**：仅偏置可置零；权重绝不可全零
+ **缺陷**：对称问题 → 所有神经元前向 / 反向完全相同，深度网络退化为线性层
+ **结论**：❌ 权重禁用，✅ 偏置可用（b=0 或小常数）

#### 2. 随机正态初始化（Random Normal）
+ **做法**：W ~ N (0, σ²)，b 通常置 0
+ **常用 σ**：0.01 ~ 0.1
+ **问题**：σ 过小→梯度消失；σ 过大→梯度爆炸；深层网络层间方差累积

#### 3. 随机均匀初始化（Random Uniform）
+ **做法**：W ~ U(-a, a)
+ **a 常取**：0.01 ~ 0.1
+ **特点**：有界、无长尾，但与正态分布相比缺乏尾部信息

#### 4. Xavier / Glorot 初始化
+ **提出**：Glorot & Bengio, 2010
+ **目标**：前向激活方差 = 反向梯度方差，各层尺度稳定
+ **正态版：**W=sqrt（2/fall_in）
+ **均匀版：**W=sqrt（6/fall_in）
+ **适用**：tanh、sigmoid、线性激活
+ **局限**：ReLU 负半轴为 0，实际方差减半 → 深层易衰减

#### 5. He / Kaiming 初始化
+ **提出**：He et al., 2015（ResNet 论文）
+ **目标**：补偿 ReLU 负半轴截断导致的方差减半
+ **正态版：**W=sqrt（2/fall_in+fall_out）
+ **均匀版：**W=sqrt（6/fall_in+fall_out）
+ **适用**：ReLU、Leaky ReLU、PReLU



### 神经网络的搭建流程：
#### 1.定义一个类继承nn.Module：
```python
class modelDemo(nn.module):
    def _init_(self):
        super()._init_()
```

#### 2.在_init_()方法中，搭建神经网络：
```python
#搭建隐藏层+输出层
#隐藏层1：输入为3，输出为3
self.linear1 = nn.Linear(in_features:3,out_features:3)

#隐藏层2：输入为3，输出为2
self.linear2 = nn.Linear(in_features:3,out_features:2)

#输出层：输入为2，输出为2
self.output = nn.Linear(in_features:2,out_features:2)

#参数初始化
#隐藏层1
nn.init.kaiming_normal_(self.linear1.weight)
nn.init.zeros_(self.linear1.bias)
#隐藏层2
nn.init.kaiming_normal_(self.linear2.weight)
nn.init.zeros_(self.linear2.bias)
```

#### 3.在forward（）方法中完成前向传播：
```python
#函数名称必须为forward，在实例化模型时会自动调用该函数
def forward(self,x):
    #第一隐藏层计算：加权求和+激活函数
    #分解版写法
    x = self.linea1(x)    #实现加权求和
    x = torch.sigmoid(x)   #激活函数
    #合并版写法
    x =torch.sigmoid(self.linear1(x))

    #第二隐藏层计算：加权求和+激活函数
    x =torch.sigmoid(self.linear2(x))

    #输出层计算：加权求和+激活函数
    x = torch.softmax(self.output(x),dim=-1)

    return x
```

#### 4.模型训练：
```python
def train():
    #创建模型对象
    model = modelDemo()
    #加载数据集
    data =torch.randn(size=(5,3),requires_grad=ture)
    #训练
    output = model(x)

    #查看模型参数
    summary(model,input_size(5,3))
```
