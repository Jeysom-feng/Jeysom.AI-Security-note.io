# Adversarial Robustness Toolbox (ART) 使用手册

> 版本基线：ART 1.17.x（Linux Foundation AI & Data 旗下 Trusted-AI 项目，原 IBM Research 开源）
> 官方文档：https://adversarial-robustness-toolbox.readthedocs.io
> 代码仓库：https://github.com/Trusted-AI/adversarial-robustness-toolbox

---

## 一、ART 是什么

ART 是一个面向**机器学习安全（ML Security）**的 Python 库，用于评估、防御、认证和验证 ML 模型面对四类对抗威胁的能力：

| 威胁类型 | 英文 | 说明 |
|---|---|---|
| 逃逸攻击 | Evasion | 推理时修改输入（对抗样本）使模型误判 |
| 投毒攻击 | Poisoning | 训练时污染数据（后门、标签翻转） |
| 模型提取 | Extraction | 通过查询复刻一个等价模型（模型盗版） |
| 推断攻击 | Inference | 成员推理、属性推理、模型反演（隐私泄露） |

**支持框架**：TensorFlow v1/v2、Keras、PyTorch、MXNet、scikit-learn、XGBoost、LightGBM、CatBoost、GPy；实验性 JAX。
**支持任务**：图像分类、目标检测（Faster R-CNN / YOLO）、语音识别（DeepSpeech / Espresso / Lingvo）、回归、GAN 生成、目标跟踪、认证鲁棒性（随机平滑 / IBP / DeepZ）。
**支持数据**：图像、表格、音频、视频。

---

## 二、安装与环境

### 2.1 环境要求
- Python **3.10 / 3.11 / 3.12**
- pip ≥ 21
- （可选）CUDA GPU 加速
- （可选）venv / conda 虚拟环境

### 2.2 安装方式

**方式一：PyPI 直接安装（推荐新手）**
```bash
# 核心包（不含深度学习框架依赖）
pip install adversarial-robustness-toolbox

# 带 PyTorch 支持
pip install adversarial-robustness-toolbox[pytorch]

# 带 TensorFlow 支持
pip install adversarial-robustness-toolbox[tensorflow]

# 带 Keras 支持
pip install adversarial-robustness-toolbox[keras]

# 全量（含开发/测试依赖）
pip install adversarial-robustness-toolbox[all]
```

**方式二：源码安装（参与开发或用最新特性）**
```bash
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
cd adversarial-robustness-toolbox
python -m venv art_env && source art_env/bin/activate   # Windows: art_env\Scripts\activate
pip install -e .[pytorch]          # 按需选 extras
```

### 2.3 验证安装
```bash
python -c "import art; print(art.__version__)"
# 输出例如 1.17.0
```

### 2.4 关键环境变量

| 变量 | 作用 | 默认值 |
|---|---|---|
| `ART_DATA_PATH` | 数据集缓存目录 | `~/.art/data` |
| `TENSORBOARD_LOGDIR` | TensorBoard 日志目录 | `./runs` |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU 卡号 | 全部可见 |

---

## 三、核心架构：四大支柱

ART 的所有功能围绕四个抽象层组织，数据流为：

```
原始模型(任意框架)
      ↓ 包装
Estimator（统一接口：predict / fit / loss_gradient）
      ↓ 被消费
Attack（生成对抗样本 / 投毒数据 / 提取模型 / 推断隐私）
      ↓ 被挂载或训练
Defence（preprocessor / postprocessor / trainer / detector）
      ↓ 量化
Metrics（经验鲁棒性、CLEVER、损失敏感度等）
```

### 3.1 Estimator（模型包装器）
把任意框架的模型包成 ART 统一接口，是所有攻击/防御的前置条件。
常用类：
- `art.estimators.classification.PyTorchClassifier`
- `art.estimators.classification.TensorFlowV2Classifier`
- `art.estimators.classification.KerasClassifier`
- `art.estimators.classification.scikitlearn.*`（SVC / RandomForest / LogisticRegression 等）
- `art.estimators.object_detection.PyTorchObjectDetector`（YOLO / Faster R-CNN）
- `art.estimators.speech_recognition.PyTorchSpeechRecognizer`
- `art.estimators.certification.randomized_smoothing.PyTorchRandomizedSmoothing`

### 3.2 Attack（攻击）
按威胁分四个子包：
- `art.attacks.evasion` —— 60+ 种逃逸攻击（白盒/黑盒）
- `art.attacks.poisoning` —— 后门、标签翻转、特征碰撞
- `art.attacks.extraction` —— Copycat CNN、Knockoff Nets
- `art.attacks.inference` —— 成员推理、属性推理、模型反演、数据库重建

### 3.3 Defence（防御）
五类防御，可组合挂载：
- `art.defences.preprocessor` —— 输入侧：特征挤压、JPEG 压缩、空间平滑、CutMix/Mixup、温度计编码
- `art.defences.postprocessor` —— 输出侧：高置信度过滤、高斯噪声、反向 Sigmoid、四舍五入
- `art.defences.trainer` —— 训练侧：对抗训练（Madry PGD / TRADES / AWP / Fast-is-Better-than-Free）
- `art.defences.detector.evasion` —— 逃逸检测器：二进制输入/激活检测、子集扫描
- `art.defences.detector.poison` —— 投毒检测器：激活聚类、谱签名、RONI、Neural Cleanse、STRIP
- `art.defences.transformer` —— 模型变换：防御性蒸馏

### 3.4 Metrics（评估指标）
`art.metrics`：经验鲁棒性（Empirical Robustness）、CLEVER 分数、损失敏感度（Loss Sensitivity）、Wasserstein 距离、成员隐私风险（SHAPr）、差分隐私训练点级度量。

---

## 四、快速上手（10 行跑通第一个对抗样本）

以 PyTorch + MNIST + FGSM 为例：

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.utils import load_mnist

# 1. 加载数据
(x_train, y_train), (x_test, y_test), min_v, max_v = load_mnist()
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test  = np.transpose(x_test,  (0, 3, 1, 2)).astype(np.float32)

# 2. 定义一个简单 CNN
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*14*14, 10)
)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. 用 ART 包装模型
classifier = PyTorchClassifier(
    model=model,
    loss=loss,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(min_v, max_v)     # 输入像素范围，非常关键
)
classifier.fit(x_train, y_train, batch_size=128, nb_epochs=3)

# 4. 干净样本准确率
acc_clean = np.mean(np.argmax(classifier.predict(x_test), 1) == np.argmax(y_test, 1))
print(f"Clean accuracy: {acc_clean:.2%}")

# 5. 构造 FGSM 攻击并生成对抗样本
attack = FastGradientMethod(estimator=classifier, eps=0.15)
x_adv = attack.generate(x=x_test)

# 6. 对抗样本准确率
acc_adv = np.mean(np.argmax(classifier.predict(x_adv), 1) == np.argmax(y_test, 1))
print(f"Adversarial accuracy (FGSM eps=0.15): {acc_adv:.2%}")
```

**输出预期**：干净准确率 ~97%，对抗准确率会掉到很低（几%到二十几%，取决于 eps）。

---

## 五、Estimator 包装器：必填参数速查

以最常用的 `PyTorchClassifier` 为例：

| 参数 | 类型 | 说明 |
|---|---|---|
| `model` | `torch.nn.Module` | 你的 PyTorch 模型 |
| `loss` | 损失函数 | 如 `nn.CrossEntropyLoss()` |
| `optimizer` | 优化器 | 如 `optim.Adam(...)` |
| `input_shape` | tuple | 单样本形状，如 `(1,28,28)` 或 `(3,32,32)` |
| `nb_classes` | int | 类别数 |
| `clip_values` | tuple | `(min, max)`，如 `(0.0, 1.0)` 或 `(0, 255)` |
| `channels_first` | bool | True=NCHW，False=NHWC；PyTorch 默认 True |
| `preprocessing_defences` | list | 挂载输入侧防御 |
| `postprocessing_defences` | list | 挂载输出侧防御 |
| `preprocessing` | tuple | `(mean, std)` 标准化，ART 会自动融入梯度计算 |

**Scikit-learn 包装器**直接传已训练好的模型即可，不需要 loss/optimizer：
```python
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from sklearn.svm import SVC
sk_model = SVC().fit(X_train, y_train)
classifier = ScikitlearnSVC(model=sk_model, clip_values=(0, 1))
```

---

## 六、常用攻击速查表

### 6.1 逃逸攻击（Evasion）白盒常用

| 攻击 | 类名 | 范数 | 特点 |
|---|---|---|---|
| FGSM | `FastGradientMethod` | L∞ | 单步最快，入门必跑 |
| BIM / I-FGSM | `BasicIterativeMethod` | L∞ | FGSM 迭代版 |
| PGD | `ProjectedGradientDescent` | L∞/L2 | **最强一阶攻击**，对抗训练标配；有 PyTorch/TF2/Numpy 三个后端 |
| C&W L∞ | `CarliniLInfMethod` | L∞ | 优化目标强，迁移性好 |
| C&W L2 | `CarliniL2Method` | L2 | 经典论文攻击 |
| DeepFool | `DeepFool` | L2 | 最小扰动估计 |
| JSMA | `SaliencyMapMethod` | L0 | 只改少量像素 |
| AutoAttack | `AutoAttack` | L∞/L2 | **集成攻击**（APGD+AE），鲁棒性评测金标准 |
| Elastic Net | `ElasticNet` | L1+L2 | 弹性网攻击 |
| Wasserstein | `Wasserstein` | Wasserstein | 基于最优传输 |
| Universal Perturbation | `UniversalPerturbation` | L∞ | 一个扰动骗所有图 |
| Adversarial Patch | `AdversarialPyTorch` / `AdversarialTensorFlowV2` | — | 贴一个真实贴纸就误识别 |
| Spatial Transformation | `SpatialTransformationAttack` | — | 平移/旋转/缩放扰动 |

**PGD 关键参数**：
```python
ProjectedGradientDescent(
    estimator=classifier,
    eps=0.3,           # 扰动预算上限
    eps_step=0.01,     # 每步步长
    max_iter=40,      # 最大迭代
    targeted=False,    # True=定向攻击，需要传 y
    num_random_init=5 # 随机重启次数
)
```

### 6.2 黑盒逃逸攻击

| 攻击 | 类名 | 场景 |
|---|---|---|
| Boundary Attack | `BoundaryAttack` | 只靠输入/输出，从零开始扰动 |
| HopSkipJump | `HopSkipJump` | 黑盒query高效 |
| Square Attack | `SquareAttack` | 随机搜索，query极少 |
| Zoo | `ZooAttack` | 零阶梯度估计 |
| Simple Black-Box | `SimpleBlackBoxAttack` | 迁移攻击 |
| Sign-OPT | `SignOPTAttack` | 基于梯度估计 |

### 6.3 投毒攻击（Poisoning）

| 攻击 | 类名 | 说明 |
|---|---|---|
| Backdoor / BadNets | `BackdoorAttack` | 植入触发器，触发时输出指定标签 |
| Clean-Label Backdoor | `CleanLabelBackdoorAttack` | 干净标签后门，更难检测 |
| Hidden Trigger Backdoor | `HiddenTriggerBackdoorAttack` | 隐藏触发器 |
| Feature Collision | `FeatureCollisionAttack` | 特征碰撞投毒 |
| Gradient Matching | `GradientMatchingAttack` | 梯度匹配（Witches' Brew） |
| Sleeper Agent | `SleeperAgentAttack` | 休眠代理（只支持 PyTorch） |
| Poisoning SVM | ` PoisoningAttackSVM` | 针对 SVM 的投毒 |

### 6.4 模型提取与推断

| 攻击 | 类名 | 目的 |
|---|---|---|
| Copycat CNN | `CopycatCNN` | 用输出标签训练替身模型 |
| Knockoff Nets | `KnockoffNets` | 主动选择查询样本提取 |
| FEE | `FunctionallyEquivalentExtraction` | 功能等价提取 |
| 成员推理（黑盒） | `MembershipInferenceBlackBox` | 判断某样本是否在训练集 |
| 成员推理（标签黑盒） | `MembershipInferenceLabelOnlyDecisionBoundary` | 只看 top-1 标签 |
| 属性推理 | `AttributeInferenceBlackBox` | 推断训练样本敏感属性 |
| 模型反演 | `MIFace` | 重构训练人脸 |
| 数据库重建 | `DatabaseReconstruction` | 重建训练数据 |

---

## 七、常用防御速查表

### 7.1 预处理防御（挂在 estimator 的 `preprocessing_defences`）

| 防御 | 类名 | 原理 |
|---|---|---|
| 特征挤压 | `FeatureSqueezing` | 降低像素位深/颜色位深 |
| JPEG 压缩 | `JpegCompression` | JPEG 有损压缩去扰动 |
| 空间平滑 | `SpatialSmoothing` | 中值/均值滤波 |
| 高斯数据增强 | `GaussianAugmentation` | 训练时加噪 |
| 温度计编码 | `ThermometerEncoding` | 量化编码 |
| Total Variance Minimization | `TotalVarMin` | 全变分去噪 |
| CutMix / Cutout / Mixup | 同名类 | 数据增强 |
| PixelDefend / DefenseGAN / InverseGAN | 同名类 | 用生成器投影回数据流形 |
| MP3 压缩 | `Mp3Compression` | 音频去扰动 |

### 7.2 后处理防御（挂在 `postprocessing_defences`）

| 防御 | 类名 | 原理 |
|---|---|---|
| 高置信度 | `HighConfidence` | 低置信度输出拒识 |
| 高斯噪声 | `GaussianNoise` | logits 加噪 |
| 反向 Sigmoid | `ReverseSigmoid` | 软化 logits |
| Rounded | `Rounded` | 量化输出 |
| 类别标签 | `ClassLabels` | 只输出硬标签 |

### 7.3 对抗训练（最有效、最常用）

```python
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerMadryPGD
from art.attacks.evasion import ProjectedGradientDescent

attack = ProjectedGradientDescent(classifier, eps=0.3, eps_step=0.01, max_iter=40)
trainer = AdversarialTrainer(classifier=classifier, attacks=attack, ratio=0.5)
# ratio=0.5 表示一半 batch 用对抗样本，一半用原始样本
trainer.fit(x_train, y_train, nb_epochs=20, batch_size=128)
```

内置对抗训练变体：
- `AdversarialTrainerMadryPGD` —— Madry 论文标准训练
- `AdversarialTrainerTRADESPyTorch` —— TRADES
- `AdversarialTrainerAWPPyTorch` —— 对抗权重扰动
- `AdversarialTrainerFastIsBetterThanFreePyTorch` —— Free 系列
- `AdversarialTrainerCertifiedIBPPyTorch` —— IBP 区间传播认证训练
- `DPInstaHideTrainer` —— 差分隐私 + InstaHide

### 7.4 投毒/后门检测

```python
from art.defences.detector.poison import ActivationDefence, SpectralSignatureDefense

defense = ActivationDefence(classifier, x_train, y_train)
report, is_clean = defense.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
poisoned_idx = np.where(is_clean == 0)[0]
```

其他检测器：`RejectOnNegativeImpact`、`NeuralCleanse`、`STRIP`、`DataProvenance`。

---

## 八、三大典型工作流

### 工作流 A：评估模型鲁棒性（白盒攻击基准）

```
训练好模型 → 包成 estimator → 同时跑 FGSM/PGD/AutoAttack → 对比干净/对抗准确率
```

```python
attacks = {
    "FGSM":  FastGradientMethod(classifier, eps=0.3),
    "PGD":   ProjectedGradientDescent(classifier, eps=0.3, eps_step=0.01, max_iter=40),
    "Auto":  AutoAttack(estimator=classifier, eps=0.3),
}
for name, atk in attacks.items():
    x_adv = atk.generate(x_test)
    acc = np.mean(np.argmax(classifier.predict(x_adv), 1) == np.argmax(y_test, 1))
    print(f"{name}: {acc:.2%}")
```

### 工作流 B：对抗训练提升鲁棒性

```
包好 estimator → 用 PGD 作为训练攻击 → AdversarialTrainer.fit → 再用 AutoAttack 评测
```

### 工作流 C：后门检测与清洗

```
疑似被投毒的模型 → ActivationDefence / NeuralCleanse → 定位触发器和脏样本 → 剔除后重训
```

### 工作流 D：成员推理隐私审计

```python
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
mi = MembershipInferenceBlackBox(estimator=classifier)
mi.fit(x_train_in, y_train_in, x_train_out, y_train_out)  # in/out 代表训练集/非训练集
membership = mi.infer(x_test, y_test)  # 1=在训练集，0=不在
```

---

## 九、关键 API 约定

**所有攻击对象的核心方法**：
- `generate(x, y=None)` —— 生成对抗样本；定向攻击时 `y` 传目标 one-hot 标签
- `set_params(**kwargs)` —— 动态改 eps / max_iter 等
- `estimator` —— 绑定的目标模型包装器

**所有 estimator 的核心方法**：
- `predict(x, batch_size=128)` —— 前向推理
- `fit(x, y, batch_size, nb_epochs, ...)` —— 训练
- `loss_gradient(x, y)` —— 对输入求损失梯度（白盒攻击底层）
- `get_activations(x, layer)` —— 取中间层激活
- `input_shape` / `nb_classes` / `clip_values` —— 属性

**数据流约定**：
- 图像默认 NHWC，PyTorch 需要 `channels_first=True` 或自己 transpose
- 标签默认 one-hot（`art.utils.to_categorical` 可转）
- 所有扰动都自动被 `clip_values` 裁剪回合法范围

---

## 十、常见坑与排错

| 问题 | 原因 / 解决 |
|---|---|
| `ImportError: cannot import name ...` | 装了核心包但没装对应框架 extras，`pip install adversarial-robustness-toolbox[pytorch]` |
| 对抗样本准确率和干净准确率一样 | `clip_values` 没设或设错；输入已经是 [0,1] 却传 (0,255) |
| PGD 跑出来比 FGSM 还差 | `eps_step` 太大导致震荡；改成 `eps/10` 左右 |
| PyTorch 报 shape mismatch | NCHW/NHWC 搞混；`channels_first` 参数要对 |
| GPU 显存爆炸 | 减小 `batch_size`；先在子集上调试 |
| 定向攻击不生效 | `targeted=True` 且 `generate(x, y=y_target)` 必须传目标标签 |
| 黑盒攻击 query 次数爆 | 选 Square/HopSkipJump 这类低 query 攻击 |
| 防御挂了等于没挂 | 确认 `apply_fit=True`（训练时生效）和 `apply_predict=True`（推理时生效） |

---

## 十一、推荐学习路径（按 4 周节奏）

1. **第 1 周 入门**：跑通 `examples/get_started_pytorch.py` 和 `mnist_cnn_fgsm.py`；把 eps 从 0.05 调到 0.5，看准确率曲线。
2. **第 2 周 攻击**：对比 FGSM / PGD / C&W / AutoAttack 在同一模型上的攻击成功率；读 `art/attacks/evasion/fast_gradient.py` 和 `projected_gradient_descent.py` 源码。
3. **第 3 周 防御**：跑 AdversarialTrainer，再用 AutoAttack 评测；试 FeatureSqueezing / JpegCompression 预处理防御。
4. **第 4 周 专题**：选一个方向深入 —— 投毒后门（BadNets + Neural Cleanse）、成员推理隐私审计、随机平滑认证鲁棒性。

**官方 Notebook 集合**：仓库 `notebooks/` 目录下有 80+ 个 Jupyter 示例，覆盖所有攻击/防御模块，是最好的参考。

---

## 十二、资源链接

- 官方文档（API 全量）：https://adversarial-robustness-toolbox.readthedocs.io
- 官网首页：https://adversarial-robustness-toolbox.org
- GitHub 仓库：https://github.com/Trusted-AI/adversarial-robustness-toolbox
- 论文（v1.0）：https://arxiv.org/abs/1807.01069
- 许可证：MIT
