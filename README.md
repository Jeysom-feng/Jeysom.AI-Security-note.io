# 🛡️ Jeysom.AI-Security-Notes

> 江西理工大学 · 信息安全专业 · 2024级
> 个人学习笔记与配套代码，聚焦 **机器学习 / 深度学习 / AI 安全（对抗样本攻防）**

## 📌 简介

这是我在系统学习 AI 安全过程中整理的笔记与代码仓库，内容涵盖机器学习基础、深度学习理论，以及对抗样本生成与防御方向的论文阅读和实践记录。笔记整理自日常学习，代码为复现与实验过程中的手写实现，用于沉淀知识、方便日后回顾。

## 📚 目录结构

```
.
├── Machine_Learning/                  # 机器学习基础
│   └── code/Regressor/               # 回归等手写实现（如 normal_equation.py）
├── Deep_Learning/                     # 深度学习
│   └── code/data/                     # 训练数据（如 train.csv）
└── adversarial_testing_training/      # 对抗测试与训练（攻击 + 防御）
    ├── code/adversarial_testing_training/
    │   ├── PGD_AT.py                  # PGD 对抗训练
    │   └── ART_Adversarial_Robustness_Toolbox_使用手册.md
    └── code/data/                     # 对抗实验数据
```

## 🗂️ 内容分类

| 方向 | 内容 |
| --- | --- |
| 🤖 机器学习 | 特征工程、数据预处理、模型训练流程、监督/无监督/半监督学习、回归手写实现 |
| 🧠 深度学习 | 网络结构、反向传播、训练技巧与数据集 |
| ⚔️🛡️ 对抗测试与训练 | FGSM/PGD 等攻击、PGD-AT / TRADES 等对抗训练、ART 工具箱使用、鲁棒性实验 |

## 💻 代码说明

各主题目录下的 `code/` 为学习过程中的配套代码：
- `Machine_Learning/code/`：sklearn 建模流程与算法手写实现
- `Deep_Learning/code/`：深度学习训练代码与数据
- `adversarial_testing_training/code/`：对抗样本生成、PGD 对抗训练及 ART 工具箱复现

> 代码多为学习复现用途，部分参数与实现以对应笔记中的实验记录为准。

## 🗺️ 学习路线

Web 安全 / Linux / 渗透 → 机器学习 & 深度学习基础 → 对抗样本攻击 → 对抗防御 → AI 安全综合实践

## 👋 关于我

- 🏫 江西理工大学 · 信息安全专业 · 2024级
- 🎯 专业方向：信息安全、网络攻防、AI 安全
- 💻 技术兴趣：Web 安全、Linux 系统、机器学习、深度学习
- 🌱 学习状态：正在探索信息安全世界的奥秘

---

*本仓库持续更新中，笔记内容难免有误，欢迎交流指正。*
