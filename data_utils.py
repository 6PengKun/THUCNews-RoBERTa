import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import numpy as np
import os # 引入 os 模块用于获取文件名

class NewsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length, label_map=None, is_test=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        self.label_map = label_map if label_map is not None else {}
        self.idx_to_label = {}
        self.texts, self.labels = self.load_data(data_path)

        # 构建或更新 idx_to_label 映射
        if self.label_map:
             self.idx_to_label = {idx: label for label, idx in self.label_map.items()}

    def load_data(self, data_path):
        texts = []
        labels_text = [] # 先存储文本标签
        build_map = not bool(self.label_map) # 只有当 self.label_map 为空时才构建 (即首次加载训练集时)

        print(f"Loading data from {data_path}...")
        # 使用 tqdm 显示文件名，更清晰
        with open(data_path, 'r', encoding='utf-8') as f:
            # 使用 os.path.basename 获取文件名用于 tqdm 描述
            for line in tqdm(f, desc=f"Reading {os.path.basename(data_path)}"):
                line = line.strip()
                if not line:
                    continue

                if self.is_test:
                    # 测试集只有文本
                    text = line
                    label = "未知" # 占位符
                else:
                    # 期望格式是 "文本\t标签"
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        # --- 正确的赋值 ---
                        text, label = parts
                        # 检查分割后文本或标签是否为空
                        if not text or not label:
                            print(f"Warning: Skipped line due to empty text or label after splitting: '{line[:100]}...'") # 只显示部分过长行
                            continue
                    else:
                        # 如果分割失败，记录警告并跳过该行
                        print(f"Warning: Skipped line due to unexpected format (expected 'text\\tlabel'): '{line[:100]}...'")
                        continue # 跳过格式不正确的行

                texts.append(text)
                labels_text.append(label)

                # 如果需要构建映射，并且标签有效且未见过
                if build_map and label != "未知" and label not in self.label_map:
                    self.label_map[label] = len(self.label_map)

        # --- 将文本标签转换为索引 ---
        label_indices = []
        num_labels_in_map = len(self.label_map)
        unknown_labels_found = set()

        for label in labels_text:
            if label in self.label_map:
                label_indices.append(self.label_map[label])
            else:
                # 对于测试集或验证/测试集中未在训练集出现的标签，设为 -1
                label_indices.append(-1)
                # 只在非构建地图模式（dev/test）且标签不是"未知"时记录未找到的标签
                if not build_map and label != "未知":
                    unknown_labels_found.add(label)

        if unknown_labels_found:
             print(f"Warning: Labels found in {data_path} but not in the provided label_map: {unknown_labels_found}")

        print(f"Loaded {len(texts)} samples.")
        if build_map:
            print(f"Built label map with {len(self.label_map)} unique labels.") # 不打印整个 map，可能太长
            # print(f"Label map: {self.label_map}") # 如果需要可以取消注释查看
        else:
             print(f"Used provided label map with {num_labels_in_map} labels.")

        # 在 load_data 结束时再次确保 idx_to_label 是最新的
        if self.label_map:
             self.idx_to_label = {idx: label for label, idx in self.label_map.items()}

        return texts, label_indices

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] # 这已经是整数索引了

        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 确保返回的张量符合预期
        return {
            'input_ids': encoding['input_ids'].squeeze(0),      # Shape: [max_seq_length]
            'attention_mask': encoding['attention_mask'].squeeze(0), # Shape: [max_seq_length]
            'label': torch.tensor(label, dtype=torch.long)      # Shape: [] (标量)
        }
    
    
    
def build_dataloader(config, tokenizer): # 移除非必要的 is_test 参数
    """构建数据加载器，确保使用统一的标签映射"""
    # 1. 加载训练集，并构建标签映射
    print("Building training dataloader and label map...")
    train_dataset = NewsDataset(
        config.train_path,
        tokenizer,
        config.max_seq_length,
        label_map=None, # 让训练集构建 map
        is_test=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    # 获取训练集构建的 label_map 和反向映射
    label_map = train_dataset.label_map
    idx_to_label = train_dataset.idx_to_label
    config.num_classes = len(label_map) # <--- 在这里设置 num_classes
    print(f"Number of classes detected: {config.num_classes}")


    # 2. 加载验证集，传入训练集的标签映射
    print("Building development dataloader...")
    dev_dataset = NewsDataset(
        config.dev_path,
        tokenizer,
        config.max_seq_length,
        label_map=label_map, # <--- 传入训练集的 map
        is_test=False
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False # 验证和测试通常不 shuffle
    )

    # 3. 加载测试集，传入训练集的标签映射
    print("Building test dataloader...")
    test_dataset = NewsDataset(
        config.test_path,
        tokenizer,
        config.max_seq_length,
        label_map=label_map, # <--- 传入训练集的 map
        is_test=True # 测试集特殊处理标签（设为-1）
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_dataloader, dev_dataloader, test_dataloader, label_map, idx_to_label # 返回所有内容


def calculate_class_weights(train_dataloader, num_classes, idx_to_label):
    """
    分析训练数据集并计算每个类别的权重
    
    参数:
        train_dataloader: 训练数据加载器
        num_classes: 类别数量
        idx_to_label: 类别ID到中文标签的映射字典
    
    返回:
        class_weights: 类别权重字典，包含ID、数量、比例、权重和归一化权重
    """
    print("计算类别权重...")
    
    # 初始化计数器
    label_counts = [0] * num_classes
    total_samples = 0
    
    # 统计每个类别的样本数量 - 使用tqdm显示进度条
    for batch in tqdm(train_dataloader, desc="统计类别分布"):
        labels = batch['label'].numpy()
        for label in labels:
            if 0 <= label < num_classes:  # 确保标签有效
                label_counts[label] += 1
                total_samples += 1
    
    # 计算每个类别的比例
    label_ratios = [count / total_samples for count in label_counts]
    
    # 计算倒数作为权重
    label_weights = [1.0 / ratio if ratio > 0 else 0.0 for ratio in label_ratios]
    
    # 归一化权重（可选）
    weight_sum = sum(label_weights)
    label_weights_norm = [weight / weight_sum * num_classes for weight in label_weights]
    
    # 创建结果字典
    class_weights = {
        'ID': list(range(num_classes)),
        'Count': label_counts,
        'Ratio': label_ratios,
        'Weight': label_weights,
        'Weight Norm': label_weights_norm
    }
    
    # 打印权重信息，加入中文标签
    print("\n类别权重信息:")
    print(f"{'标签':<10} {'ID':<4} {'Count':<8} {'Ratio':<8} {'Weight':<10} {'Weight Norm':<10}")
    print("-" * 55)
    for i in range(num_classes):
        label_text = idx_to_label.get(i, "未知")  # 获取中文标签，如果不存在则显示"未知"
        print(f"{label_text:<10} {i:<4} {label_counts[i]:<8} {label_ratios[i]:.6f} {label_weights[i]:.6f} {label_weights_norm[i]:.6f}")
    
    return class_weights


def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        


def validate_dataset(dataloader, label_map, max_samples=10):
    """验证数据集加载是否正确"""
    
    # 反转标签映射，用于显示原始标签
    idx_to_label = {idx: label for label, idx in label_map.items()}
    
    print("\n=== 数据集验证 ===")
    print(f"标签映射: {label_map}")
    print(f"共发现 {len(label_map)} 个标签类别")
    
    # 获取一个批次的数据
    batch = next(iter(dataloader))
    
    # 统计 batch 中的有效数据数量
    print(f"批次大小: {len(batch['input_ids'])}")
    
    # 检查张量形状
    print(f"输入ID形状: {batch['input_ids'].shape}")
    print(f"注意力掩码形状: {batch['attention_mask'].shape}")
    print(f"标签形状: {batch['label'].shape}")
    
    # 检查标签分布
    label_counts = {}
    for label in batch['label'].numpy():
        label_str = idx_to_label.get(label, f"未知({label})")
        label_counts[label_str] = label_counts.get(label_str, 0) + 1
    
    print("\n批次中的标签分布:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(batch['label'])*100:.2f}%)")
    
    # 显示几个样本
    print(f"\n显示前 {min(max_samples, len(batch['input_ids']))} 个样本:")
    for i in range(min(max_samples, len(batch['input_ids']))):
        input_ids = batch['input_ids'][i]
        attention_mask = batch['attention_mask'][i]
        label = batch['label'][i].item()
        
        # 解码文本
        tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(input_ids)
        text = dataloader.dataset.tokenizer.convert_tokens_to_string(tokens)
        
        print(f"\n样本 #{i+1}:")
        print(f"  文本: {text}")
        print(f"  标签索引: {label}")
        print(f"  标签文本: {idx_to_label.get(label, '未知')}")
        print(f"  有效Token数: {sum(attention_mask).item()}")
        print(f"  原始文本: {dataloader.dataset.texts[i]}")
    
    return True