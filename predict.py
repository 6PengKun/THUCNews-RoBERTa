import torch
import numpy as np
from tqdm import tqdm

def predict(model, dataloader, config):
    """使用模型进行预测"""
    model.eval()
    
    all_logits = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # 将数据移到指定设备
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            
            # 保存结果
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 拼接结果
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    # 获取预测类别
    preds = np.argmax(all_logits, axis=1)
    
    return preds, all_probs

def save_predictions(predictions, texts, output_file, idx_to_label):
    """保存预测结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for text, pred_idx in zip(texts, predictions):
            pred_label = idx_to_label.get(pred_idx, "未知")
            f.write(f"{text}\t{pred_label}\n")
    
    print(f"Predictions saved to {output_file}")
    

def save_predictions_(predictions, output_file, idx_to_label):
    """保存预测结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred_idx in predictions:
            pred_label = idx_to_label.get(pred_idx, "未知")
            f.write(f"{pred_label}\n")
    
    print(f"Predictions saved to {output_file}")