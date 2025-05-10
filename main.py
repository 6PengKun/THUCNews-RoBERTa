import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import argparse

from config import Config
from data_utils import build_dataloader, set_seed, validate_dataset,calculate_class_weights
from model import EnhancedLightAttentionModel
from train import train_model
from evaluate import evaluate_model,evaluate_model_final
from predict import predict, save_predictions, save_predictions_

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RoBERTa for Chinese text classification")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction")
    parser.add_argument("--resume_training", action="store_true", help="是否从断点恢复训练")
    args = parser.parse_args()
    
    # 配置信息
    config = Config()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 打印训练设置
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    # 加载分词器
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 构建 Dataloaders 并获取标签信息
    train_dataloader, dev_dataloader, test_dataloader, label_map, idx_to_label = build_dataloader(config, tokenizer)

    
#     # 验证数据集加载
#     print("\n验证训练集加载...")
#     validate_dataset(train_dataloader, label_map)
    
#     print("\n验证验证集加载...")
#     validate_dataset(dev_dataloader, label_map)
    
#     print("\n验证测试集加载...")
#     validate_dataset(test_dataloader, label_map)
    
    # 设置类别数量
    config.num_classes = len(label_map)
    print(f"Number of classes: {config.num_classes}")
    
    # 初始化模型
    print(f"Initializing RoBERTa model")
    model = EnhancedLightAttentionModel(
        config.model_name,
        config.num_classes,
        # max_seq_length=config.max_seq_length
    )
    model.to(config.device)
    
    # 将标签映射保存到config中，以便在预测时使用
    config.label_map = label_map
    config.idx_to_label = idx_to_label

    # 仅预测
    if args.do_predict and not args.do_train and not args.do_eval:
        best_model_path = os.path.join(config.save_path, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("No saved model found. Please train the model first.")
        print("Predicting on test set...")
        test_preds, test_probs = predict(model, test_dataloader, config)

        # 获取测试集文本
        test_texts = []
        with open(config.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_texts.append(line)

        # 保存预测结果
        save_predictions(
            test_preds, 
            test_texts, 
            os.path.join(config.save_path, "predictions.txt"),
            config.idx_to_label
        )
        
        save_predictions_(
            test_preds,  
            os.path.join(config.save_path, "predictions_.txt"),
            config.idx_to_label
        )

        return
    
    # 定义损失函数
    # PyTorch 版本
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    if args.do_train:
        print("开始训练...")
        model = train_model(model, train_dataloader, dev_dataloader, config, criterion=criterion, resume=args.resume_training)
    
    # 加载最佳模型
    if args.do_eval or args.do_predict:
        best_model_path = os.path.join(config.save_path, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("No saved model found. Please train the model first.")
    
    # 评估模型
    if args.do_eval:
        print("Evaluating model on dev set...")
        dev_accuracy, dev_f1, dev_loss, dev_class_metrics = evaluate_model_final(
            model, dev_dataloader, criterion, config
        )
        print(f"Dev Loss: {dev_loss:.4f}")
        print(f"Dev Accuracy: {dev_accuracy:.4f}")
        print(f"Dev F1 Score: {dev_f1:.4f}")
    
    # 预测
    if args.do_predict:
        print("Predicting on test set...")
        test_preds, test_probs = predict(model, test_dataloader, config)

        # 获取测试集文本
        test_texts = []
        with open(config.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_texts.append(line)

        # 保存预测结果
        save_predictions(
            test_preds, 
            test_texts, 
            os.path.join(config.save_path, "predictions.txt"),
            config.idx_to_label
        )
        
        save_predictions_(
            test_preds,  
            os.path.join(config.save_path, "predictions_.txt"),
            config.idx_to_label
        )

if __name__ == "__main__":
    main()