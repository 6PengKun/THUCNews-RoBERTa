# 中文新闻标题分类系统

基于RoBERTa的THUCNews数据集新闻标题分类项目，实现了14分类任务。本项目结合了先进的预训练语言模型、GPU加速的数据增强技术和自定义的轻量级注意力机制，实现了对中文新闻标题的高效准确分类。

## 📋 项目描述

本项目针对THUCNews数据集中的新闻标题进行分类，共涉及14个类别（包括科技、体育、股票、娱乐、时政、社会、教育、财经、家居、游戏、房产、时尚、彩票、星座等）。

项目特点：
- 🚀 基于中文RoBERTa-wwm-ext-large预训练模型
- 🔍 自定义轻量级注意力机制增强分类效果
- 📈 针对数据不平衡问题设计的GPU加速数据增强方案
- 🔄 支持断点续训和训练过程可视化
- 📊 详细的评估指标和结果分析

计算资源需求：
- 最低配置是10GB显存，需要缩小batch_size
- 建议是24GB显存

## 🔧 技术栈

- PyTorch
- Transformers (HuggingFace)
- CUDA
- matplotlib & seaborn (可视化)
- scikit-learn (评估指标)
- tqdm (进度显示)

## 📁 项目结构

```
.
├── data/                       # 数据目录
│   ├── dev.txt                 # 开发/验证集数据
│   ├── test.txt                # 测试集数据
│   └── train.txt               # 训练集数据
├── apply_data_augmentation.py  # GPU加速的数据增强工具
├── config.py                   # 配置文件
├── data_utils.py               # 数据加载与处理
├── evaluate.py                 # 模型评估
├── main.py                     # 主程序入口
├── model.py                    # 模型定义
├── predict.py                  # 预测功能
└── train.py                    # 模型训练
```

## ✨ 核心功能

### 数据增强

针对THUCNews数据集中类别不平衡问题，设计了基于GPU加速的数据增强方案：

- 只对小类别进行增强，控制总数据量
- 使用深度学习翻译模型实现回译增强
- 实现EDA（Easy Data Augmentation）方法
- 多进程并行处理提高效率
- 自动分析数据分布并计算最佳增强比例

### 模型设计

基于RoBERTa预训练模型，增加了：

- 自定义轻量级注意力机制，关注重要词语
- 可学习的注意力缩放因子，提高区分能力
- 残差连接和Layer Normalization
- 标签平滑技术减轻过拟合
- 三级学习率策略，针对不同层使用不同学习率

### 训练与评估

- 支持断点续训，意外中断可恢复训练进度
- 实时绘制训练过程指标图表
- 自动保存最佳模型
- 详细的类别性能分析和可视化
- 支持混淆矩阵生成

## 🚀 使用方法

### 环境准备

```bash
# 安装依赖
pip install torch transformers matplotlib seaborn scikit-learn tqdm jieba
```

### 数据增强

```bash
python apply_data_augmentation.py --input data/train.txt --output data/augmented_train.txt
```
———————————————————————————————————————————————

### 模型训练

```bash
python main.py --do_train
```

### 模型评估

```bash
python main.py --do_eval
```

### 模型预测

```bash
python main.py --do_predict
```

### 断点续训

```bash
python main.py --do_train --resume_training
```

### 完成训练、评估、预测，并且开启断点续训
```bash
python main.py --do_train --do_eval --do_predict --resume_training 
```

## 📊 实验结果

在THUCNews数据集上，本模型取得了以下性能：

不使用数据增强：
```python
# config.py
# 模型参数
model_name = "hfl/chinese-roberta-wwm-ext-large"  # 中文RoBERTa模型
num_classes = None  # 将在运行时动态设置
max_seq_length = 48  # 根据分析，99%的文本长度不超过27个字符，设置32作为安全值

# 训练参数
batch_size = 128
gradient_accumulation_steps = 2  # 梯度累积步数
num_epochs = 4
learning_rate = 1e-4
warmup_proportion = 0.1
weight_decay = 0

random_seed = 42
```
- 验证集总体准确率：99.40%
- 宏平均F1分数：99.40%
- 小类别（如彩票、星座）的F1分数相比基线提升xx.xx%
- 测试集总体准确率：89.03%

使用数据增强：
```python
# config.py
# 模型参数
model_name = "hfl/chinese-roberta-wwm-ext-large"  # 中文RoBERTa模型
num_classes = None  # 将在运行时动态设置
max_seq_length = 48  # 根据分析，99%的文本长度不超过27个字符，设置32作为安全值

# 训练参数
batch_size = 128
gradient_accumulation_steps = 2  # 梯度累积步数
num_epochs = 5  # 训练完4个epoch后发现验证集上准确率很高，实际上只训练了4个epoch
learning_rate = 1e-4
warmup_proportion = 0.2
weight_decay = 0

random_seed = 42
```
- 验证集总体准确率：99.48%
- 宏平均F1分数：99.54%
- 小类别（如彩票、星座）的F1分数相比基线提升xx.xx%
- 测试集总体准确率：89.15%

注：由于该任务训练集和验证集的相似程度较高，但是在测试集表现提升很难，似乎存在一个泛化边界（更换不同的基座，总是达到相近的测试集性能），复杂化的特征提取反而效果较差，在有限的计算资源下，很难做出飞跃性的提升。
89.15%是目前尝试的所有常规架构（预训练模型基座 + 顶层的特征变换 + 分类层）中最好的单模性能，并且架构相对简单。
进一步的提升需要考虑模型集成和直推式学习。



## 🔍 核心创新点

1. **智能数据增强策略**：根据类别分布自动计算每个类别的增强比例，着重增强小类别
2. **GPU加速的回译增强**：使用GPU加速翻译模型进行回译，速度提升10倍以上
3. **轻量级注意力机制**：融合CLS表示和注意力加权表示，捕捉标题中的关键信息
4. **可视化训练监控**：实时生成训练指标图表，直观展示训练进度和模型性能

## 📝 参考资料

- THUCNews数据集: [链接](https://aistudio.baidu.com/datasetdetail/103654/0)
- Chinese RoBERTa-wwm-ext-large: [链接](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)


## 👨‍💻 作者

[Peng Kun](https://github.com/6PengKun)
