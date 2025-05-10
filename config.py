import torch

class Config:
    # 数据参数
    train_path = "../data/augmented_train.txt"
    dev_path = "../data/dev.txt"
    test_path = "../data/test.txt"
    
    # 模型参数
    model_name = "hfl/chinese-roberta-wwm-ext-large"  # 中文RoBERTa模型
    num_classes = None  # 将在运行时动态设置
    max_seq_length = 48  # 根据分析，99%的文本长度不超过27个字符，设置32作为安全值
    
    # 训练参数
    batch_size = 128
    gradient_accumulation_steps = 2  # 梯度累积步数
    num_epochs = 5
    learning_rate = 1e-4
    warmup_proportion = 0.2
    weight_decay = 0
    
    # 其他
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./saved_model"
    random_seed = 42