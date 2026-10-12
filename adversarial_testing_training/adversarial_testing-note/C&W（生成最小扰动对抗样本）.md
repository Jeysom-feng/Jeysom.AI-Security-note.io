

### 原理：
#### 优化目标：

$$

\min_{\delta}\ \|\delta\|_2^2 + c\cdot f(x+\delta)
\quad \text{s.t. } x+\delta\in[\min,\max]

$$

$$f(x')=\max\!\left(\max_{i\neq t} Z(x')_i - Z(x')_t,\ -\kappa\right)

$$

$$x+\delta=\frac{1}{2}\big(\tanh(w)+1\big)(\max-\min)+\min
\quad\Longrightarrow\quad
\delta(w)=\frac{1}{2}\big(\tanh(w)+1\big)(\max-\min)+\min-x

$$

δ表示修改像素的量，c是平衡扰动大小与攻击成功率的参数；

寻找最小扰动的过程就是使这个优化目标函数最小的过程；

当扰动过大时，δ的二范数会比较大，即使cf（x+δ）是负数成功使模型判断错误，但是整体还是比较大的正数

当扰动过小时，f（x+δ）>0表示攻击未成功

所以C&W就是要找到一个c与δ使目标函数值最小

#### f（x）的理解：
第一项算出其他类的最大logit-目标类的logit，然后再与-k比较

k这个参数是信心参数，若目标类的logit已经很高了，那个f（x）的输出会被截至到k而不会无限制增加，也就是其他类logit与目标类logit的差距到底要多大才能确信攻击成功

### 过程：
先选定一个c的值例如c=1，再随机选择一个w，使用Adam梯度下降不断调整也就是调正δ使得优化目标函数值下降，在经过指定轮数后，看f（x）输出的值是否>0；如果>0说明攻击未成功，调大c进行测试

若f（x）输出的值<0,使用更小的c继续测试，

:::color4
c太小： f 项没话语权，loss 被||δ||² 主导，w 很快收敛到 δ≈0—收敛了但收敛到失败的解

c太大： f 项权重压过一切，优化器猛冲成功方向。f 一旦变负，loss 里就只剩 ||δ||²-ck，反而好收敛 —— 但收敛出来的 δ 很大（噪声大到变了张图）收敛得快，但解不好。  

**c的调整使用二分法： 从大 c 开始（稳成功），逐步缩小 c，c 小到某次内层怎么跑都失败时，说明到临界点了 —— 退一步用上一次成功的 c，那个 δ 就是最优解  **

:::

#### 伪代码示例：
```plain
内层：
c 取一个值（比如 c=10）
从随机 w 开始，Adam 迭代 1000 步：
    每步都在调 w → 也就是在调 δ
    算 loss = ||δ||² + 10·f(x+δ)
    梯度下降让 loss 变小
收敛后得到一个 δ*
外层：
猜 c=1   → 内层优化出 δ → 检查 f(δ)<0 了吗？没成功 → c 太小，加大
猜 c=10  → 内层优化出 δ → 成功了，但 ||δ||² 太大 → c 太大，减小
猜 c=5   → 内层重新优化出 δ → 没成功 → c 还是小
猜 c=7   → 内层重新优化出 δ → 成功了，且 ||δ||²=0.05 很小 → 就它了

```



#### 基于CIFAR-10数据集的对抗测试（用ART实现）：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ud6d4455b" class="ne-image">

##### 代码：
```python
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
```

### 实验结果：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791551250937-b0634af9-3210-4873-9e4d-dbd136a2698e.png" width="1624.107135942548" alt="" title="原样本预测准确率为：55.79%，对抗样本预测准确率：18.30%" crop="0,0,1,1" id="uf33a6427" class="ne-image">




