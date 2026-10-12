### 核心工作原理
它不依赖真实梯度，核心思路是**边界游走 + 蒙特卡洛梯度近似 + 二分精修**：通过随机采样扰动的分类结果，近似估计决策边界的法向量（梯度方向），再沿着边界切向移动并不断向原始点收紧，逐步逼近距离原始样本最近的边界点

### 完整执行流程
#### 1. 初始化：Hop 阶段
生成一个大扰动的初始样本，确保它落在对抗区域（被误分类），作为迭代起点：

+ 无目标攻击：对原始样本加随机噪声，直到分类偏离真实类别；
+ 定向攻击：生成属于目标类别的样本，或大扰动强制分类为目标类别。

#### 2. 迭代三步循环（每轮迭代）
##### 第一步：Jump — 二分搜索逼近决策边界
在**原始样本 ****↔**** 当前对抗样本**的连线上做二分搜索，精准定位到刚好落在决策边界上的临界点（刚好触发误分类的最小扰动点）。这一步直接把当前方向上的扰动收紧到最小值。

##### 第二步：Skip — 蒙特卡洛估计边界法向量
在边界点周围施加一批随机方向的小扰动，根据这些扰动的分类结果（是否仍在对抗侧），用蒙特卡洛方法近似计算决策边界的法向量（即梯度方向）。采样批量随迭代次数按根号t递增，前期快速探索、后期提升精度。

##### 第三步：Hop — 沿边界切向前进一步
沿着边界的切向（垂直于法向量方向）移动一步，得到新的对抗样本，进入下一轮迭代。通过沿边界不断游走，逐步探索到距离原始点更近的边界位置。

### 核心特性
1. **最弱黑盒假设**：仅需要模型输出的分类标签，是对模型访问权限要求最低的攻击之一，适合只能拿到分类结果的云端 API 场景。
2. **最小扰动导向**：原生针对L2范数优化，最终生成的扰动非常接近理论最小值，扰动质量优于同查询量级的其他黑盒攻击。
3. **查询效率优秀**：比传统有限差分近似梯度的黑盒攻击少 1~2 个数量级的查询次数，是决策基黑盒攻击中效率最高的代表算法。
4. **自适应精度调度**：采样批量随迭代逐步增大，步长逐步缩小，平衡搜索效率和最终精度。



#### 基于CIFAR-10数据集的对抗测试：
##### 模型结构：
<img src="https://cdn.nlark.com/yuque/0/2026/png/62806879/1791802809428-9295ce75-5659-49ad-a5a6-7a94c1ae8aa1.png" width="613.3928545313527" alt="" title="" crop="0,0,1,1" id="ud6d4455b" class="ne-image">

##### 代码：
```python
def HopSkipJump(model,test_dataset):
     model.eval()
     test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
     total_sample,adv_correct,clean_correct=0,0,0
     attack = HopSkipJumpAttack(
    model,
    norm='L2',       # 原生主支持 L2 范数
    eps=0.3,         # 最大扰动约束
    steps=50,        # 迭代轮数
    n_queries=2000,  # 最大查询次数上限
    init_size=0.1,   # 初始扰动大小
    batch_size=64    # 蒙特卡洛采样批量
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
