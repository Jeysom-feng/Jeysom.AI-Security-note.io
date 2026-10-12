AutoAttack：集成评估套件，4 种攻击组合，用来可靠评估鲁棒性

### APGD-CE:
#### 原理与过程：
 初始步长默认设为2ε（最大扰动约束），保证攻击初期可以在可行域内快速大范围探索，当交叉熵损失函数值增长不明显或出现震荡时，判断是步长过大导致，使步长乘以缩减系数 ρ（默认为0.75）；

 每次步长缩减后，迭代会从历史最优扰动点重新开始，避免步长缩小后轨迹偏离最优区域  

##### 多轮重启机制（辅助参数探索）
+ 算法支持多次随机重启（由 `n_restarts` 控制），每次重启从全新的随机初始点开始搜索，避免单轮迭代陷入局部最优。
+ 步长缩减后的 “内部重启” 与随机重启互相配合，既保证精细搜索的深度，又覆盖全局探索的广度。

#### 针对PGD做的改进：
##### 1. 解决了「步长需手动调参、高度依赖人工经验」的易用性问题
标准 PGD 的迭代步长、学习率(α）是固定值，需要人工根据扰动约束ε、目标模型、迭代次数手动调试：

+ 步长过大：迭代过程震荡发散，无法收敛到有效扰动；
+ 步长过小：收敛速度极慢，有限迭代次数内攻击强度不足。  
且不存在通用最优步长，更换模型、ε或数据集后都需要重新调参，使用门槛高、复现性差。

##### APGD-CE 的解决方案：
通过损失趋势反馈 + 震荡检测实现步长全自动自适应：初始设为2ε的大步长快速探索，当损失增长停滞或出现震荡时，自动按系数ρ（默认 0.75）缩减步长，全程无需人工设置步长，可自动适配不同的攻击场景，实现 “免调参” 攻击。

##### 2. 解决了「固定步长导致前期收敛慢、后期易震荡」的收敛效率问题
标准 PGD 的固定步长天然存在矛盾：

+ 若步长适配前期大范围探索，后期就会在最优解附近来回震荡，无法收敛到更优的局部极小值；
+ 若步长适配后期精细搜索，前期探索效率极低，大量迭代次数被浪费。

##### **APGD-CE 的解决方案**：  
采用**渐进式步长缩减 + 最优点重启**的策略：
+ 前期大步长快速穿越可行域，快速逼近最优扰动区域；
+ 损失停滞时自动缩小步长，切换为局部精细搜索；
+ 每次步长缩减后，从历史最优扰动点重新开始迭代，避免步长缩小后轨迹偏离最优区域，同时兼顾了收敛速度和最终收敛精度。

##### 3. 解决了「单起点易陷入局部最优、攻击成功率上限低」的攻击强度问题
标准 PGD 通常从原始样本（零扰动）或单个随机起点开始迭代，很容易陷入局部最优解；面对防御能力较强的模型时，攻击成功率会出现明显下降。

#### 基于CIFAR-10数据集的对抗测试：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ud6d4455b" class="ne-image">

##### 代码：
```python
def APGD_CE(model,test_dataset):
     model.eval()
     attack = torchattacks.APGD(
        model, 
        norm='Linf', 
        eps=8/255, 
        steps=10,
        loss='ce'
    )
     #生成对抗样本
     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
     total_clean = 0
     total_adv = 0
     total_samples = 0
     for imgs, labels in test_loader:
        with torch.no_grad():
            clean_logits = model(imgs)
        clean_pred = torch.argmax(clean_logits, dim=1)
        # 生成对抗样本
        adv_imgs = attack(imgs, labels)
        with torch.no_grad():
            adv_logits = model(adv_imgs)
        adv_pred = torch.argmax(adv_logits, dim=1)
        total_clean += (clean_pred == labels).sum().item()
        total_adv += (adv_pred == labels).sum().item()
        total_samples += imgs.shape[0]
     acc_clean = total_clean / total_samples
     acc_adv = total_adv / total_samples
     print(f"Clean accuracy: {acc_clean:.2%}")
     print(f"adv accuracy: {acc_adv:.2%}")
```

##### 测试结果：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791632104853-3f6db99c-3abe-46b1-a3c2-eea23c34cba2.png" width="1608.928564578599" alt="" title="" crop="0,0,1,1" id="u63cdffeb" class="ne-image">

干净样本准确率：55.79%

对抗样本准确率：0.04%

### APGD-DLR:
#### 优化目标：
$ $\mathcal{L}_{\mathrm{DLR}}(x,y) = -\dfrac{z_{y}(x) - \max\limits_{i \neq y} z_{i}(x)}{z_{\pi_{1}}(x) - z_{\pi_{3}}(x)}
 $

#### 相比 APGD-CE 的核心改进（解决的问题）
#### DLR 损失专门针对交叉熵损失的三大固有缺陷设计：
##### 1. 解决了 CE 损失对 logit 平移 / 缩放敏感、易被防御失效的问题
交叉熵损失的数值和梯度会随 logit 整体平移、整体缩放发生显著变化，面对 logit 压缩、梯度掩码、温度缩放等防御手段时，CE 损失的梯度会被弱化甚至消失，攻击极易失效。

DLR 损失具备平移不变性和缩放不变性：分子和分母均为 logit 的差值，整体平移或缩放 logit 不会改变损失值和梯度方向，因此对各类基于 logit 变换的防御手段鲁棒性更强，是 AutoAttack 中单攻击变体里抗防御能力最稳定的一个。

##### 2. 解决了 CE 损失低置信度下梯度饱和、无法深入攻击的问题
当真实类别的置信度已经被压得很低时，交叉熵损失的梯度会趋近于 0（梯度饱和），无法继续推动优化，攻击容易停在 “分类错误但间隔很小” 的浅度误分类状态。

DLR 直接优化真实类与最强错误类的 logit 差值，即使真实类概率已经极低，只要分类间隔还能拉大，梯度就会持续存在，能够不断向决策边界深处推进，攻击强度更高，尤其适合对抗鲁棒性较强的模型。

##### 3. 解决了 CE 损失优化目标与攻击最终目标不一致的问题
交叉熵优化的是概率分布的散度，目标是 “让真实类概率尽可能低”，但这并不等价于 “让模型稳定分错”—— 有时 CE 损失数值很高，但分类间隔依然很小，微小的扰动就会回到正确分类。

DLR 直接瞄准分类决策的核心指标 ——分类间隔，最大化真实类与次优类的 logit 差距，优化目标与 “让模型可靠误分类” 的攻击最终目标完全一致，生成的对抗样本误分类更稳定、置信度更高。



#### 基于CIFAR-10数据集的对抗测试：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ZZauD" class="ne-image">

##### 代码：
```python
def APGD_DLR(model,test_dataset):
    model.eval()
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
    total_sample,adv_correct,clean_correct=0,0,0
    #构造APGD-DLR对象
    attack = torchattacks.APGD(
            model, 
            norm='Linf', 
            eps=8/255, 
            steps=100,
            loss='dlr',
            n_restarts=3
        )
    for x,y in test_loader:
        x_adv = attack(x,y)
        with torch.no_grad():
            adv_pred =torch.argmax(model(x_adv),dim=1) 
        adv_correct += (adv_pred == y).sum().item()
        with torch.no_grad():
            clean_pred = torch.argmax(model(x),dim=1)
        clean_correct += (clean_pred == y).sum().item()
        total_sample += x.size(0)
        acc_clean = clean_correct / total_sample
        acc_adv = adv_correct / total_sample
    print(f"Clean accuracy: {acc_clean:.2%}")
    print(f"adv accuracy: {acc_adv:.2%}")
```

##### 测试结果：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791638660356-83b98130-5f13-4663-8e80-367c76fb8a71.png" width="1625.8928502206595" alt="" title="" crop="0,0,1,1" id="ua851d050" class="ne-image">

干净样本准确率：55.79%

对抗样本准确率：0.00%

### FAB-T（依据决策边界找最小扰动）:
 AB-T 全称 **Fast Adaptive Boundary Attack - Targeted**（快速自适应边界攻击 - 有目标版本），是 AutoAttack 对抗攻击评估套件中的**白盒定向攻击算法**。它的核心目标是在指定的Lp 范数约束下，找到**扰动幅度最小**的对抗样本，使得模型将输入样本精准错误分类为攻击者指定的目标类别  

####  过程：
在距离原始样本 min{u,ε} /2的位置初始化迭代起点（ε为扰动范数上限）  ，每次通过向决策边界进行投影逐渐靠近目标类与原始类的决策边界，

Xproj更新公式为：

L2范数更新：

$ \boldsymbol{x}_{\text{proj}} = \boldsymbol{x}_t - \frac{g(\boldsymbol{x}_t)}{\left\| \nabla g(\boldsymbol{x}_t) \right\|_2^2} \cdot \nabla g(\boldsymbol{x}_t)
 $：

L∞范数更新：

$ \boldsymbol{x}_{\text{proj}} = \boldsymbol{x}_t - \frac{g(\boldsymbol{x}_t)}{\left\| \nabla g(\boldsymbol{x}_t) \right\|_1} \cdot \operatorname{sign}\left( \nabla g(\boldsymbol{x}_t) \right)
 $

L1范数更新：

$ \boldsymbol{x}_{\text{proj}} = \boldsymbol{x}_t - \frac{g(\boldsymbol{x}_t)}{\left| \nabla g(\boldsymbol{x}_t)_{k^*} \right|} \cdot \boldsymbol{e}_{k^*}
\quad \text{其中} \quad
k^* = \arg\max_{i} \left| \nabla g(\boldsymbol{x}_t)_i \right|
 $

更新后与Xt做加权求和得出下一迭代点

$ \boldsymbol{x}_{t+1} = \alpha_t \cdot \boldsymbol{x}_{\text{proj}} + (1 - \alpha_t) \cdot \boldsymbol{x}_t
 $

在迭代结束后， 原始样本 X0 与当前最优对抗样本Xbest 之间做线性插值，通过二分搜索定位到刚好越过决策边界的最小扰动点  

$ \boldsymbol{x}^* = \beta^* \cdot \boldsymbol{x}_0 + (1 - \beta^*) \cdot \boldsymbol{x}_{\text{best}}

\text{其中} \quad \beta^* = \max\left\{ \beta \in [0,1] \mid g\left( \beta \boldsymbol{x}_0 + (1-\beta) \boldsymbol{x}_{\text{best}} \right) > 0 \right\}
 $



#### 基于CIFAR-10数据集的对抗测试：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="l5OiS" class="ne-image">

##### 代码：
```python
def FAB(model,test_dataset):
     model.eval()
     test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
     total_sample,adv_correct,clean_correct=0,0,0
        #构造FAB对象
     attack = torchattacks.FAB(
                model, 
                norm='Linf', 
                eps=8/255, 
                steps=15,
            #   multi_targeted=True 
            )
     for x,y in test_loader:
            x_adv = attack(x,y)
            with torch.no_grad():
                adv_pred =torch.argmax(model(x_adv),dim=1) 
            adv_correct += (adv_pred == y).sum().item()
            with torch.no_grad():
                clean_pred = torch.argmax(model(x),dim=1)
            clean_correct += (clean_pred == y).sum().item()
            total_sample += x.size(0)
            acc_clean = clean_correct / total_sample
            acc_adv = adv_correct / total_sample
     print(f"Clean accuracy: {acc_clean:.2%}")
     print(f"adv accuracy: {acc_adv:.2%}")
```

##### 测试结果：
###### 定向攻击：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791683727528-88fb5d55-96e0-4c20-a46b-5ce55844c846.png" width="1627.6785644987713" alt="" title="" crop="0,0,1,1" id="uac3990ee" class="ne-image">

干净样本准确率：55.79%

对抗样本准确率：1.36%

###### 无定向攻击：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791683828373-af7b2467-0e74-4a3f-a576-a12820e50d00.png" width="1225.8928519236495" alt="" title="" crop="0,0,1,1" id="u8d2d35a6" class="ne-image">

干净样本准确率：55.79%

对抗样本准确率：0.00%

### Square Attack:
#### 核心执行流程
##### 1. 初始化
+ 用满足 L∞约束的随机扰动生成初始对抗样本，确保初始状态攻击成功（样本被误分类）；若初始失败则放大扰动直至成功。
+ 初始方形块设为较大尺寸（通常为图像边长的 1/2），对应全局粗粒度搜索。

##### 2. 随机采样方形扰动
每一步随机生成候选扰动：

+ 随机位置：在图像平面内随机选择方形的左上角坐标
+ 随机大小：采用当前迭代阶段对应的方形边长
+ 随机取值：对方形内的每个像素，在 （-η，η） 范围内随机修改像素值，全程保证不超出 ε的最大扰动约束

##### 3. 评估与更新准则
将候选样本输入模型查询预测结果，按以下规则更新：

+ 若候选样本**攻击成功且扰动范数更小**：直接接受，替换当前最优样本
+ 若候选样本**攻击成功但扰动更大**：以一定概率接受（避免陷入局部最优）
+ 若攻击失败：直接拒绝，保留当前最优解

##### 4. 迭代精细化
随着迭代次数增加，逐步缩小方形块的边长，从全局粗搜索过渡到局部像素级微调，最终收敛到接近理论最优的最小扰动。



#### 方形窗口更新规则：
| 采样次数区间 | 窗口面积比例 p | 窗口大小特点 |
| --- | --- | --- |
| 0 ~ 10 次 | 初始最大值 | 最大窗口，全局粗搜索 |
| 10 ~ 50 次 | 初始值的 1/2 | 次大窗口 |
| 50 ~ 200 次 | 初始值的 1/4 | 中等窗口 |
| 200 ~ 1000 次 | 初始值的 1/8 | 中偏小窗口 |
| 1000 ~ 2000 次 | 初始值的 1/16 | 小窗口 |
| 2000 次之后 | 持续逐级减半 | 逐步收缩到像素级微调 |


#### 基于CIFAR-10数据集的对抗测试：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ezmvd" class="ne-image">

##### 代码：
```python
def Square_Attack(model,test_dataset):
         model.eval()
         test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
         total_sample,adv_correct,clean_correct=0,0,0
            #构造SquareAttack对象
         attack =Square(
                    model, 
                    norm='Linf', 
                    eps=8/255, 
                    n_queries=10000,
                    n_restarts = 1
                )
         for x,y in test_loader:
                x_adv = attack(x,y)
                with torch.no_grad():
                    adv_pred =torch.argmax(model(x_adv),dim=1) 
                adv_correct += (adv_pred == y).sum().item()
                with torch.no_grad():
                    clean_pred = torch.argmax(model(x),dim=1)
                clean_correct += (clean_pred == y).sum().item()
                total_sample += x.size(0)
                acc_clean = clean_correct / total_sample
                acc_adv = adv_correct / total_sample
         print(f"Clean accuracy: {acc_clean:.2%}")
         print(f"adv accuracy: {acc_adv:.2%}")
```

##### 测试结果：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791686232029-60112fde-512b-41d0-92e4-1adb9ee3b966.png" width="1609.8214217176546" alt="" title="" crop="0,0,1,1" id="u07a5527e" class="ne-image">

干净样本准确率：55.79%

对抗样本准确率：0.00%
