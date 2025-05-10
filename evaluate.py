import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, dataloader, criterion, config, final_eval=False, save_cm=False):
    """
    评估模型，支持计算每个类别的性能指标
    
    参数:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        config: 配置参数
        final_eval: 是否是最终评估
        save_cm: 是否保存混淆矩阵
    
    返回:
        accuracy: 准确率
        f1: 宏平均F1分数
        avg_loss: 平均损失
        class_metrics: 每个类别的指标(如果final_eval=True)
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            # 将数据移到指定设备
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # 保存结果
            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 拼接结果
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 获取预测类别
    preds = np.argmax(all_logits, axis=1)
    
    # 计算总体指标
    accuracy = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='macro')
    
    # 如果是最终评估，计算每个类别的指标
    if final_eval:
        # 每个类别的性能指标
        class_precision = precision_score(all_labels, preds, average=None, zero_division=0)
        class_recall = recall_score(all_labels, preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, preds, average=None, zero_division=0)
        
        # 计算样本数量
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        sample_counts = {label: count for label, count in zip(unique_labels, counts)}
        all_classes_sample_counts = {i: sample_counts.get(i, 0) for i in range(config.num_classes)}
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, preds, labels=range(config.num_classes))
        
        # 计算每个类别的准确率
        class_accuracy = []
        for i in range(config.num_classes):
            # 类别i预测正确的样本数除以类别i的总样本数
            if cm[i].sum() > 0:
                class_accuracy.append(cm[i, i] / cm[i].sum())
            else:
                class_accuracy.append(0.0)
        
        # 创建每个类别的指标字典
        class_metrics = {}
        for i in range(config.num_classes):
            class_name = config.idx_to_label[i]
            class_metrics[class_name] = {
                "样本数量": all_classes_sample_counts[i],
                "准确率": class_accuracy[i],
                "精确率": class_precision[i],
                "召回率": class_recall[i],
                "F1分数": class_f1[i]
            }
        
        # 如果需要保存混淆矩阵
        if save_cm:
            plot_confusion_matrix(cm, config.idx_to_label, config.save_path)
        
        return accuracy, f1, avg_loss, class_metrics
    
    return accuracy, f1, avg_loss

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix with English labels"""
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set heatmap
    ax = sns.heatmap(
        cm_normalized, 
        annot=False,  # Too many classes, don't show numbers
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Adjust label font size
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=45, fontsize=10, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300)
    print(f"Confusion matrix saved to {os.path.join(save_path, 'confusion_matrix.png')}")
    plt.close()

def plot_class_metrics(class_metrics, save_path):
    """Plot class metrics comparison with English labels"""
    class_names = list(class_metrics.keys())
    precision_values = [metrics['精确率'] for metrics in class_metrics.values()]
    recall_values = [metrics['召回率'] for metrics in class_metrics.values()]
    f1_values = [metrics['F1分数'] for metrics in class_metrics.values()]
    sample_counts = [metrics['样本数量'] for metrics in class_metrics.values()]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # Plot three metrics bars
    ax.bar(x - width, precision_values, width, label='Precision', color='#5DA5DA')
    ax.bar(x, recall_values, width, label='Recall', color='#FAA43A')
    ax.bar(x + width, f1_values, width, label='F1 Score', color='#60BD68')
    
    # Add sample count annotations
    for i, count in enumerate(sample_counts):
        ax.text(i, -0.05, f"n={count}", ha='center', va='top', fontsize=8, rotation=45)
    
    # Set chart properties
    ax.set_title('Performance Metrics by Class', fontsize=16)
    ax.set_ylabel('Metric Value', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_metrics.png'), dpi=300)
    print(f"Class metrics chart saved to {os.path.join(save_path, 'class_metrics.png')}")
    plt.close()

def evaluate_model_final(model, dataloader, criterion, config):
    """最终评估模型，包含详细的类别性能分析"""
    print("正在进行最终模型评估...")
    
    # 执行评估，获取类别指标
    accuracy, f1, avg_loss, class_metrics = evaluate_model(
        model, dataloader, criterion, config, final_eval=True, save_cm=True
    )
    
    # 打印总体指标
    print("\n===== 总体性能 =====")
    print(f"准确率: {accuracy:.4f}")
    print(f"宏平均F1分数: {f1:.4f}")
    print(f"损失: {avg_loss:.4f}")
    
    # 打印各类别性能
    print("\n===== 各类别性能 =====")
    
    # 创建性能表格
    headers = ["类别", "样本数", "准确率", "精确率", "召回率", "F1分数"]
    row_format = "{:<15} {:<8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}"
    
    print(" ".join([f"{h:<15}" if i == 0 else f"{h:<8}" for i, h in enumerate(headers)]))
    print("-" * 60)
    
    for class_name, metrics in sorted(class_metrics.items(), key=lambda x: x[1]['F1分数']):
        print(row_format.format(
            class_name[:15], 
            metrics['样本数量'],
            metrics['准确率'],
            metrics['精确率'],
            metrics['召回率'],
            metrics['F1分数']
        ))
    
    # 绘制类别指标对比图
    plot_class_metrics(class_metrics, config.save_path)
    
    # 保存详细指标到文件
    save_metrics_to_file(class_metrics, accuracy, f1, avg_loss, config.save_path)
    
    return accuracy, f1, avg_loss, class_metrics

def save_metrics_to_file(class_metrics, accuracy, f1, avg_loss, save_path):
    """将评估指标保存到文件"""
    with open(os.path.join(save_path, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write("===== 总体性能 =====\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"宏平均F1分数: {f1:.4f}\n")
        f.write(f"损失: {avg_loss:.4f}\n\n")
        
        f.write("===== 各类别性能 =====\n")
        headers = ["类别", "样本数", "准确率", "精确率", "召回率", "F1分数"]
        f.write(" ".join([f"{h:<15}" if i == 0 else f"{h:<8}" for i, h in enumerate(headers)]) + "\n")
        f.write("-" * 60 + "\n")
        
        # 按F1分数排序
        sorted_metrics = sorted(class_metrics.items(), key=lambda x: x[1]['F1分数'], reverse=True)
        
        for class_name, metrics in sorted_metrics:
            f.write(f"{class_name[:15]:<15} {metrics['样本数量']:<8} {metrics['准确率']:<8.4f} "
                   f"{metrics['精确率']:<8.4f} {metrics['召回率']:<8.4f} {metrics['F1分数']:<8.4f}\n")
    
    print(f"评估结果已保存至 {os.path.join(save_path, 'evaluation_results.txt')}")