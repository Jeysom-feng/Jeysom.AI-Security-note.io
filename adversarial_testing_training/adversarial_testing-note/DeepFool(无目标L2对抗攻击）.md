

### 公式：

$$

\begin{align*}
d &= \frac{|F|}{\|w\|_2} \\
\boldsymbol{r}_i &= \frac{|f'_l|}{\|w_l\|_2^2} \boldsymbol{w}_l \\
F_k(\boldsymbol{x}) &= f_k(\boldsymbol{x}) - f_{\hat{k}_0}(\boldsymbol{x}) \\
\boldsymbol{w}_k &= \nabla f_k(\boldsymbol{x}) - \nabla f_{\hat{k}_0}(\boldsymbol{x}) \\
\hat{l}&=\arg\min_{k\neq \hat{k}_0}\frac{|F_k(\boldsymbol{x}_i)|}{\|\boldsymbol{w}_k\|_2} \\
\boldsymbol{x}_{\text{adv}} &= \boldsymbol{x}_0 + (1+\epsilon)\sum_{i} \boldsymbol{r}_i
\end{align*}

$$

#### 对公式的理解：
每次迭代计算最小扰动ri,ri通过公式：点xi到决策边界的距离乘以边界法向量得出，最终对抗样本由ri进行累加得出

#### 基于CIFAR-10数据集的对抗测试：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ud6d4455b" class="ne-image">

##### 代码：
```python
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
```

### 测试结果：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791621709087-5c787211-8e11-447c-9169-2f2a7483dbeb.png" width="1610.7142788567105" alt="" title="" crop="0,0,1,1" id="u579552c6" class="ne-image">

干净样本准确率：55.79%

对抗样本准确率：12.24%
