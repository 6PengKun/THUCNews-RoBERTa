import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from evaluate import evaluate_model

def label_smoothing_loss(logits, labels, smoothing=0.1):
    """标签平滑化损失函数"""
    confidence = 1.0 - smoothing
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_f1, best_epoch, metrics, path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'best_dev_f1': best_f1,
        'best_epoch': best_epoch,
        'metrics': metrics  # 保存训练指标历史
    }, path)
    
def save_training_info(save_dir, info_dict):
    """保存训练信息到JSON文件"""
    with open(os.path.join(save_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4)

def plot_training_metrics(epochs, train_losses, dev_losses, dev_accuracies, dev_f1s, save_path):
    """Plot training and validation metrics charts with highlighted data points"""
    plt.figure(figsize=(15, 15))
    
    # Plot loss curves with markers
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', markersize=6)
    plt.plot(epochs, dev_losses, 'ro-', label='Validation Loss', markersize=6)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Plot accuracy curve with markers
    plt.subplot(3, 1, 2)
    plt.plot(epochs, dev_accuracies, 'go-', label='Validation Accuracy', markersize=6)
    plt.title('Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Plot F1 score curve with markers
    plt.subplot(3, 1, 3)
    plt.plot(epochs, dev_f1s, 'co-', label='Validation F1 Score', markersize=6)
    plt.title('Validation F1 Score', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate best performance points
    if dev_losses:
        best_loss_epoch = epochs[dev_losses.index(min(dev_losses))]
        best_loss = min(dev_losses)
        plt.subplot(3, 1, 1)
        plt.scatter(best_loss_epoch, best_loss, s=100, c='red', marker='*', 
                    label='Best Loss', zorder=5)
        plt.annotate(f'Best: {best_loss:.4f}', 
                    (best_loss_epoch, best_loss),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontweight='bold')
        plt.legend(fontsize=12)
    
    if dev_accuracies:
        best_acc_epoch = epochs[dev_accuracies.index(max(dev_accuracies))]
        best_acc = max(dev_accuracies)
        plt.subplot(3, 1, 2)
        plt.scatter(best_acc_epoch, best_acc, s=100, c='green', marker='*', 
                    label='Best Accuracy', zorder=5)
        plt.annotate(f'Best: {best_acc:.4f}', 
                    (best_acc_epoch, best_acc),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontweight='bold')
        plt.legend(fontsize=12)
    
    if dev_f1s:
        best_f1_epoch = epochs[dev_f1s.index(max(dev_f1s))]
        best_f1 = max(dev_f1s)
        plt.subplot(3, 1, 3)
        plt.scatter(best_f1_epoch, best_f1, s=100, c='cyan', marker='*', 
                    label='Best F1', zorder=5)
        plt.annotate(f'Best: {best_f1:.4f}', 
                    (best_f1_epoch, best_f1),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontweight='bold')
        plt.legend(fontsize=12)
    
    # Save the chart with higher DPI for better quality
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=300)
    print(f"Training metrics chart saved to {os.path.join(save_path, 'training_metrics.png')}")
    plt.close()

def train_model(model, train_dataloader, dev_dataloader, config, criterion=None, resume=False):
    """
    训练模型，支持断点续训和指标可视化
    
    参数:
        model: 模型
        train_dataloader: 训练数据加载器
        dev_dataloader: 验证数据加载器
        config: 配置参数
        criterion: 损失函数
        resume: 是否从断点恢复训练
    
    返回:
        训练好的模型
    """
    # 确保保存目录存在
    os.makedirs(config.save_path, exist_ok=True)
    
    # 初始化训练状态和记录指标的容器
    start_epoch = 0
    global_step = 0
    best_dev_f1 = 0
    best_epoch = 0
    train_losses = []
    dev_losses = []
    dev_accuracies = []
    dev_f1s = []
    epochs = []
    
    # 检查点路径
    checkpoint_path = os.path.join(config.save_path, "checkpoint.pt")
    
    # 优化器设置，对预训练层和分类层使用不同的学习率
    # 使用三组不同的学习率
    params = [
        {"params": model.roberta.parameters(), "lr": config.learning_rate / 2},  # 预训练模型使用较小学习率
        {"params": [model.attention.weight, model.attention.bias, model.attention_scale], 
         "lr": config.learning_rate},  # 注意力机制使用较大学习率
        {"params": [p for n, p in model.named_parameters() 
                   if not any(nd in n for nd in ['roberta', 'attention.weight', 'attention.bias', 'attention_scale'])], 
         "lr": config.learning_rate}  # 其他层使用标准学习率
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)
    
    # 学习率调度器 - 使用余弦退火
    total_steps = (len(train_dataloader) // config.gradient_accumulation_steps) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_proportion),
        num_training_steps=total_steps
    )
    
    # 如果未传入损失函数，使用默认的交叉熵损失
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    # 如果需要恢复训练
    if resume and os.path.exists(checkpoint_path):
        print(f"正在从检查点恢复训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复学习率调度器状态
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        global_step = checkpoint['global_step']
        best_dev_f1 = checkpoint['best_dev_f1']
        best_epoch = checkpoint['best_epoch']
        
        # 恢复指标历史
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            train_losses = metrics.get('train_losses', [])
            dev_losses = metrics.get('dev_losses', [])
            dev_accuracies = metrics.get('dev_accuracies', [])
            dev_f1s = metrics.get('dev_f1s', [])
            epochs = metrics.get('epochs', [])
        
        print(f"已恢复到 Epoch {start_epoch}, 全局步数 {global_step}")
        print(f"当前最佳 F1: {best_dev_f1:.4f} (Epoch {best_epoch+1})")
    
    # 训练循环
    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # 训练模式
        model.train()
        total_loss = 0
        
        # 进度条
        train_bar = tqdm(train_dataloader, desc="训练中")
        
        # 重置优化器
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_bar):
            # 将数据移到指定设备
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            # 计算损失
            loss = label_smoothing_loss(logits, labels)
            # 缩放损失以进行梯度累积
            loss = loss / config.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 累计损失
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # 改为显示学习率:
            current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
            train_bar.set_postfix(
                loss=loss.item() * config.gradient_accumulation_steps,
                lr=current_lr
            )
            
            # # 基于批次步数保存检查点
            # if (step + 1) % 20000 == 0:  # 每500个批次保存一次，不受梯度累积影响
            #     # 收集当前指标
            #     metrics = {
            #         'train_losses': train_losses,
            #         'dev_losses': dev_losses,
            #         'dev_accuracies': dev_accuracies,
            #         'dev_f1s': dev_f1s,
            #         'epochs': epochs
            #     }
            #     save_checkpoint(model, optimizer, scheduler, epoch, global_step, 
            #                   best_dev_f1, best_epoch, metrics, checkpoint_path)
            #     print(f"已保存检查点：批次 {step+1}")
            
            # 检查是否需要更新参数
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新参数
                optimizer.step()
                scheduler.step()
                
                # 重置梯度
                optimizer.zero_grad()
                
                # 增加全局步数
                global_step += 1
        
        # 确保最后一个批次的梯度也得到更新
        if len(train_dataloader) % config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # 计算平均损失并记录
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        epochs.append(epoch + 1)
        print(f"平均训练损失: {avg_train_loss:.4f}")
        
        
        dev_accuracy, dev_f1, dev_loss = evaluate_model(
            model, dev_dataloader, criterion, config
        )
        
        # 记录验证指标
        dev_losses.append(dev_loss)
        dev_accuracies.append(dev_accuracy)
        dev_f1s.append(dev_f1)
        
        print(f"验证损失: {dev_loss:.4f}")
        print(f"验证准确率: {dev_accuracy:.4f}")
        print(f"验证F1分数: {dev_f1:.4f}")
        
        # 保存当前 epoch 的检查点
        metrics = {
            'train_losses': train_losses,
            'dev_losses': dev_losses,
            'dev_accuracies': dev_accuracies,
            'dev_f1s': dev_f1s,
            'epochs': epochs
        }
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, 
                      best_dev_f1, best_epoch, metrics, checkpoint_path)
        
        # 绘制当前进度的指标图
        plot_training_metrics(epochs, train_losses, dev_losses, dev_accuracies, dev_f1s, config.save_path)
        
        # 保存最佳模型
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            
            # 保存模型
            torch.save(model.state_dict(), os.path.join(config.save_path, "best_model.pt"))
            print(f"保存新的最佳模型，F1分数: {dev_f1:.4f}")
            
            # 保存训练信息
            save_training_info(config.save_path, {
                'best_epoch': best_epoch,
                'best_f1': best_dev_f1,
                'current_epoch': epoch,
                'total_epochs': config.num_epochs,
                'global_step': global_step
            })
    
    print(f"训练完成。最佳F1分数: {best_dev_f1:.4f} (Epoch {best_epoch+1})")
    
    # 绘制最终指标图
    plot_training_metrics(epochs, train_losses, dev_losses, dev_accuracies, dev_f1s, config.save_path)
    
    return model