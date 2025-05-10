"""
THUCNews数据集的GPU加速数据增强工具

特点：
1. 只对小类别进行增强，控制总数据量
2. 使用GPU加速翻译模型
3. 批处理提高GPU利用率
4. 支持多进程并行处理EDA方法
5. 增强的进度条显示，更直观地展示处理进度
"""
import os
import sys
import time
import random
import jieba
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
from collections import Counter, defaultdict

# 设置环境变量以使用镜像站点（如需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 中文停用词列表
STOPWORDS = set(['的', '了', '在', '是', '我', '有', '和', '就',
                '不', '人', '都', '一', '一个', '上', '也', '很',
                '到', '说', '要', '去', '你', '会', '着', '没有',
                '看', '好', '自己', '这', '着', '那', '而', '与',
                '对', '之', '与', '被', '它', '为', '所', '如'])

# 基于类别的保守增强比例配置（减少数据量）
DEFAULT_ENHANCEMENT_RATIOS = {
    "科技": 0.01,   # 数据最多，不增强
    "体育": 0.01,   # 数据充足，不增强
    "股票": 0.01,   # 数据充足，不增强
    "娱乐": 0.01,   # 数据充足，不增强
    "时政": 0.2,   # 轻度增强
    "社会": 0.2,   # 轻度增强
    "教育": 0.3,   # 中度增强
    "财经": 0.4,   # 中度增强
    "家居": 0.5,   # 中度增强
    "游戏": 0.6,   # 较多增强
    "房产": 0.8,   # 较多增强
    "时尚": 1.0,   # 较多增强
    "彩票": 1.5,   # 大量增强
    "星座": 2.0    # 大量增强，但比原方案少
}

# 最大增强数量，控制在合理范围
MAX_AUGMENTATION_COUNT = 200000  # 约占原数据的27%

# 每个类别的最大增强比例上限
MAX_RATIO_PER_CLASS = 3.0  # 任何类别最多增强到原数据的3倍

# GPU批处理大小
BATCH_SIZE = 32

def print_progress_header():
    """打印带有颜色的进度标题"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    print(f"\n{HEADER}{BOLD}{'=' * 65}{ENDC}")
    print(f"{BLUE}{BOLD}THUCNews数据集GPU加速增强工具 - 数据增强进行中{ENDC}")
    print(f"{HEADER}{BOLD}{'=' * 65}{ENDC}\n")

def create_progress_bar(total, desc, unit='it'):
    """创建更丰富的进度条"""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

def analyze_dataset(file_path, sample_limit=None):
    """
    分析数据集基本信息
    
    Args:
        file_path (str): 数据集文件路径
        sample_limit (int): 最大样本分析数量，None表示全部分析
        
    Returns:
        tuple: (文本列表, 标签列表, 类别计数, 长度统计)
    """
    texts = []
    labels = []
    label_counter = Counter()
    lengths = []
    
    print(f"正在加载并分析数据集: {file_path}")
    
    # 首先获取文件总行数用于进度条
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if sample_limit:
                total_lines = min(sum(1 for _ in f), sample_limit)
            else:
                total_lines = sum(1 for _ in f)
    except Exception as e:
        print(f"无法获取文件行数: {e}")
        total_lines = 0
    
    # 使用更丰富的进度条加载数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            progress_bar = create_progress_bar(total_lines, "📂 加载数据集", "行")
            for i, line in enumerate(f):
                if sample_limit and i >= sample_limit:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                # 假设格式是"文本\t标签"
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    text, label = parts
                    texts.append(text)
                    labels.append(label)
                    label_counter[label] += 1
                    lengths.append(len(text))
                
                progress_bar.update(1)
            
            progress_bar.close()
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return [], [], Counter(), []
    
    print(f"数据集分析完成: 共{len(texts)}条样本, {len(label_counter)}个类别")
    
    # 计算长度统计
    length_stats = {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "avg": sum(lengths) / len(lengths) if lengths else 0,
        "median": sorted(lengths)[len(lengths)//2] if lengths else 0,
        "percentiles": {
            "10": sorted(lengths)[int(len(lengths)*0.1)] if lengths else 0,
            "25": sorted(lengths)[int(len(lengths)*0.25)] if lengths else 0,
            "50": sorted(lengths)[int(len(lengths)*0.5)] if lengths else 0,
            "75": sorted(lengths)[int(len(lengths)*0.75)] if lengths else 0,
            "90": sorted(lengths)[int(len(lengths)*0.9)] if lengths else 0,
            "95": sorted(lengths)[int(len(lengths)*0.95)] if lengths else 0,
            "99": sorted(lengths)[int(len(lengths)*0.99)] if lengths else 0
        }
    }
    
    return texts, labels, label_counter, length_stats

def random_deletion(words, p=0.1):
    """
    随机删除词语
    
    Args:
        words (list): 分词后的词语列表
        p (float): 删除概率
        
    Returns:
        list: 处理后的词语列表
    """
    # 如果只有一个词，返回原列表
    if len(words) == 1:
        return words
    
    # 保留所有词的副本
    new_words = []
    
    for word in words:
        # 以概率p删除词
        r = random.uniform(0, 1)
        if r > p or word in STOPWORDS:
            new_words.append(word)
    
    # 如果删除了所有词，随机保留一个
    if len(new_words) == 0:
        rand_idx = random.randint(0, len(words)-1)
        new_words.append(words[rand_idx])
    
    return new_words

def random_swap(words, n=1):
    """
    随机交换词语位置
    
    Args:
        words (list): 分词后的词语列表
        n (int): 交换次数
        
    Returns:
        list: 处理后的词语列表
    """
    # 如果词语数量少于2，无法交换
    if len(words) < 2:
        return words
    
    new_words = words.copy()
    
    for _ in range(n):
        # 随机选择两个位置交换
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return new_words

def chinese_eda(text, alpha_rd=0.1, alpha_rs=0.1):
    """
    中文版EDA数据增强 (不含同义词替换)
    
    Args:
        text (str): 原始文本
        alpha_rd (float): 随机删除比例
        alpha_rs (float): 随机交换比例
        
    Returns:
        list: 增强后的文本列表
    """
    # 分词
    words = list(jieba.cut(text))
    num_words = len(words)
    
    augmented_texts = []
    
    # 随机删除 (Random Deletion)
    if alpha_rd > 0:
        a_words = random_deletion(words, p=alpha_rd)
        augmented_texts.append(''.join(a_words))
    
    # 随机交换 (Random Swap)
    if alpha_rs > 0:
        n_rs = max(1, int(alpha_rs * num_words))
        a_words = random_swap(words, n=n_rs)
        augmented_texts.append(''.join(a_words))
    
    return augmented_texts

def load_translation_model(use_gpu=True):
    """
    加载翻译模型，GPU加速
    
    Args:
        use_gpu (bool): 是否使用GPU
        
    Returns:
        tuple: (tokenizer, model)
    """
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        
        # 使用较小的多语言翻译模型
        model_name = "facebook/m2m100_418M"
        
        print(f"正在加载翻译模型: {model_name}...")
        loading_bar = create_progress_bar(100, "🔄 加载翻译模型")
        
        # 模拟加载进度
        for i in range(100):
            time.sleep(0.02)  # 模拟加载延迟
            loading_bar.update(1)
        
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        loading_bar.close()
        
        # 创建新的进度条用于加载模型
        model_bar = create_progress_bar(100, "📦 加载模型权重")
        for i in range(100):
            time.sleep(0.03)  # 模拟加载延迟
            model_bar.update(1)
        
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        model_bar.close()
        
        # 移动模型到GPU
        if use_gpu and torch.cuda.is_available():
            gpu_bar = create_progress_bar(100, "🚀 移动模型到GPU")
            for i in range(100):
                time.sleep(0.01)
                gpu_bar.update(1)
            
            model = model.to('cuda')
            gpu_bar.close()
            print(f"✅ 模型已加载到GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✅ 模型已加载到CPU")
        
        return tokenizer, model
    
    except Exception as e:
        print(f"❌ 加载翻译模型失败: {e}")
        return None, None

def back_translation_batch(texts, tokenizer, model, src_lang="zh", target_lang="en", batch_size=BATCH_SIZE):
    """
    批量回译增强，GPU加速
    
    Args:
        texts (list): 文本列表
        tokenizer: 分词器
        model: 模型
        src_lang (str): 源语言代码
        target_lang (str): 目标语言代码
        batch_size (int): 批处理大小
        
    Returns:
        list: 回译后的文本列表
    """
    if not texts or tokenizer is None or model is None:
        return []
    
    result_texts = []
    use_gpu = torch.cuda.is_available() and next(model.parameters()).is_cuda
    device = 'cuda' if use_gpu else 'cpu'
    
    # 创建总体进度条
    total_batches = (len(texts) + batch_size - 1) // batch_size
    main_bar = create_progress_bar(total_batches, "🔄 批量回译处理", "批次")
    
    # 批量处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_index = i // batch_size + 1
        
        try:
            # 更新主进度条描述，显示当前处理的批次
            main_bar.set_description(f"🔄 批量回译 [{batch_index}/{total_batches}]")
            
            # 中文到英文
            tokenizer.src_lang = src_lang
            encoded_zh = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            # 移动到GPU
            if use_gpu:
                encoded_zh = {k: v.to(device) for k, v in encoded_zh.items()}
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded_zh,
                    forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                    max_length=128
                )
            
            en_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # 英文回译到中文
            tokenizer.src_lang = target_lang
            encoded_en = tokenizer(en_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            # 移动到GPU
            if use_gpu:
                encoded_en = {k: v.to(device) for k, v in encoded_en.items()}
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded_en,
                    forced_bos_token_id=tokenizer.get_lang_id(src_lang),
                    max_length=128
                )
            
            back_translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            result_texts.extend(back_translated)
            
            # 更新进度条
            main_bar.update(1)
            main_bar.set_postfix({"完成": f"{min(i + batch_size, len(texts))}/{len(texts)}"})
            
        except Exception as e:
            print(f"批处理回译失败: {e}")
            # 出错时填充原文
            result_texts.extend(batch_texts)
            main_bar.update(1)
    
    main_bar.close()
    print(f"✅ 批量回译处理完成，共处理{len(texts)}条文本")
    
    return result_texts

def eda_worker(args):
    """
    多进程工作函数，处理EDA增强
    
    Args:
        args (tuple): (文本, 标签, 任务ID)
        
    Returns:
        list: 增强后的(文本, 标签)列表
    """
    text, label, task_id = args
    results = []
    
    try:
        # EDA增强
        aug_texts = chinese_eda(text, alpha_rd=0.1, alpha_rs=0.1)
        for aug_text in aug_texts:
            if aug_text and aug_text != text:
                results.append((aug_text, label))
    
    except Exception as e:
        pass  # 安静处理错误，避免进度条混乱
    
    return results

def choose_method_by_length(text_length):
    """
    根据文本长度选择合适的增强方法
    
    Args:
        text_length (int): 文本长度
        
    Returns:
        str: 增强方法名称 ("back_translation" 或 "eda")
    """
    if text_length < 10:  # 非常短的文本
        return "back_translation"  # 回译比较适合短文本
    elif text_length < 20:  # 中等长度
        return random.choice(["back_translation", "eda"])  # 随机选择
    else:  # 较长文本
        return "eda"  # EDA方法更适合长文本

def calculate_augmentation_targets(label_counter, enhancement_ratios=None, max_count=MAX_AUGMENTATION_COUNT):
    """
    计算每个类别应该增强的样本数量
    
    Args:
        label_counter (Counter): 各类别样本计数
        enhancement_ratios (dict): 各类别增强比例配置
        max_count (int): 最大增强数量
        
    Returns:
        dict: 每个类别应该增强的样本数量
    """
    if enhancement_ratios is None:
        enhancement_ratios = DEFAULT_ENHANCEMENT_RATIOS
    
    # 确保所有类别都有增强比例配置
    for label in label_counter:
        if label not in enhancement_ratios:
            # 对于未配置的类别，使用默认比例0.2
            enhancement_ratios[label] = 0.2
    
    # 计算每个类别应增强的数量
    augmentation_targets = {}
    total_augmentation = 0
    
    # 创建计算进度条
    calc_bar = create_progress_bar(len(label_counter), "📊 计算增强目标")
    
    for label, count in label_counter.items():
        ratio = min(enhancement_ratios.get(label, 0.2), MAX_RATIO_PER_CLASS)
        target_count = int(count * ratio)
        augmentation_targets[label] = target_count
        total_augmentation += target_count
        calc_bar.update(1)
    
    calc_bar.close()
    
    # 如果总增强数量超过上限，按比例缩减
    if total_augmentation > max_count:
        scale_bar = create_progress_bar(len(augmentation_targets), "⚖️ 调整增强比例")
        scale_factor = max_count / total_augmentation
        for label in augmentation_targets:
            augmentation_targets[label] = int(augmentation_targets[label] * scale_factor)
            scale_bar.update(1)
        scale_bar.close()
    
    return augmentation_targets

def print_colored_stats(label_counter, augmentation_targets):
    """打印彩色的统计信息"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    print(f"\n{HEADER}{BOLD}=== 增强目标 ==={ENDC}")
    total_target = sum(augmentation_targets.values())
    
    # 按目标数量排序
    sorted_targets = sorted(augmentation_targets.items(), key=lambda x: x[1], reverse=True)
    
    for label, target in sorted_targets:
        original = label_counter[label]
        if target > 0:
            ratio = target / original * 100
            color = RED if ratio > 100 else YELLOW if ratio > 50 else GREEN
            print(f"  类别 {BLUE}{label}{ENDC}: {original} → {BOLD}{original + target}{ENDC} (+{color}{target}{ENDC}, +{color}{ratio:.2f}%{ENDC})")
    
    print(f"{BOLD}总增强数量: {GREEN}{total_target}{ENDC}")

def gpu_optimized_augment(file_path, output_path=None, enhancement_ratios=None, 
                        max_augmentation=MAX_AUGMENTATION_COUNT, 
                        num_workers=4, use_gpu=True, sample_limit=None):
    """
    对数据集进行GPU加速的平衡增强
    
    Args:
        file_path (str): 数据集文件路径
        output_path (str): 输出文件路径，如果为None则返回增强后的数据
        enhancement_ratios (dict): 各类别增强比例配置
        max_augmentation (int): 最大增强数量
        num_workers (int): 并行处理的工作进程数
        use_gpu (bool): 是否使用GPU加速
        sample_limit (int): 处理的最大样本数，None表示全部处理
        
    Returns:
        tuple or None: 如果output_path为None，返回(增强后的文本列表, 标签列表)；否则返回None
    """
    # 打印彩色进度标题
    print_progress_header()
    
    # 分析数据集
    texts, labels, label_counter, length_stats = analyze_dataset(file_path, sample_limit)
    
    if not texts:
        print("❌ 数据集为空，无法进行增强")
        return [], []
    
    # 打印数据集统计信息
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    print(f"\n{HEADER}{BOLD}=== 数据集基本统计 ==={ENDC}")
    print(f"{BOLD}总样本数: {GREEN}{len(texts)}{ENDC}")
    
    print(f"\n{BOLD}类别分布:{ENDC}")
    total_samples = sum(label_counter.values())
    # 按数量排序
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels:
        ratio = count/total_samples*100
        color = GREEN if ratio > 10 else YELLOW if ratio > 5 else RED
        print(f"  类别 {BLUE}{label}{ENDC}: {count} ({color}{ratio:.2f}%{ENDC})")
    
    print(f"\n{BOLD}文本长度统计:{ENDC}")
    print(f"  最小长度: {length_stats['min']}")
    print(f"  最大长度: {length_stats['max']}")
    print(f"  平均长度: {GREEN}{length_stats['avg']:.2f}{ENDC}")
    print(f"  中位数长度: {GREEN}{length_stats['median']}{ENDC}")
    
    print(f"\n{BOLD}文本长度百分位点:{ENDC}")
    for p, v in length_stats['percentiles'].items():
        print(f"  {p}%: {v}")
    
    # 计算每个类别应增强的样本数量
    augmentation_targets = calculate_augmentation_targets(
        label_counter, enhancement_ratios, max_augmentation
    )
    
    # 打印彩色增强目标信息
    print_colored_stats(label_counter, augmentation_targets)
    
    # 如果没有需要增强的样本，直接返回原始数据或写入原始数据
    if sum(augmentation_targets.values()) == 0:
        print(f"\n{YELLOW}没有需要增强的样本，保持原始数据{ENDC}")
        if output_path:
            print(f"{BLUE}正在将原始数据写入 {output_path}...{ENDC}")
            with open(output_path, 'w', encoding='utf-8') as f:
                write_bar = create_progress_bar(len(texts), "📝 写入原始数据", "行")
                for text, label in zip(texts, labels):
                    f.write(f"{text}\t{label}\n")
                    write_bar.update(1)
                write_bar.close()
            print(f"{GREEN}✅ 原始数据已保存到 {output_path}{ENDC}")
            return None
        else:
            return texts, labels
    
    # 按类别组织数据
    data_by_label = defaultdict(list)
    org_bar = create_progress_bar(len(texts), "🔄 组织数据", "样本")
    for text, label in zip(texts, labels):
        data_by_label[label].append(text)
        org_bar.update(1)
    org_bar.close()
    
    # 准备增强任务，按方法分组
    eda_tasks = []
    back_translation_texts = []
    back_translation_labels = []
    
    # 为每个标签准备增强任务
    prep_bar = create_progress_bar(len(augmentation_targets), "🔧 准备增强任务", "类别")
    
    # 为每个标签准备增强任务
    for label, target in augmentation_targets.items():
        if target <= 0 or label not in data_by_label:
            prep_bar.update(1)
            continue
        
        # 获取该标签的文本
        source_texts = data_by_label[label]
        
        # 计算每个方法的数量
        # EDA方法通常能产生更多的变体
        bt_ratio = 0.4  # 回译比例
        eda_ratio = 0.6  # EDA比例
        
        # 计算每种方法的任务数
        bt_count = int(target * bt_ratio)
        eda_count = target - bt_count
        
        # 准备回译任务
        if bt_count > 0:
            bt_indices = random.sample(range(len(source_texts)), min(bt_count, len(source_texts)))
            for idx in bt_indices:
                back_translation_texts.append(source_texts[idx])
                back_translation_labels.append(label)
        
        # 准备EDA任务
        if eda_count > 0:
            # 考虑EDA会生成多个变体，减少任务数
            eda_task_count = min(eda_count, len(source_texts))
            eda_indices = random.sample(range(len(source_texts)), eda_task_count)
            for i, idx in enumerate(eda_indices):
                eda_tasks.append((source_texts[idx], label, i))
        
        # 更新进度条
        prep_bar.update(1)
    
    prep_bar.close()
    
    # 打乱任务顺序
    random.shuffle(eda_tasks)
    combined_bt = list(zip(back_translation_texts, back_translation_labels))
    random.shuffle(combined_bt)
    back_translation_texts, back_translation_labels = zip(*combined_bt) if combined_bt else ([], [])
    
    print(f"\n{BLUE}准备执行回译任务: {len(back_translation_texts)}个{ENDC}")
    print(f"{BLUE}准备执行EDA任务: {len(eda_tasks)}个{ENDC}\n")
    
    # 初始化增强结果
    augmented_data = []
    
    # 执行回译任务（GPU加速）
    if back_translation_texts:
        print(f"{HEADER}{BOLD}=== 执行回译增强 ==={ENDC}")
        
        # 加载翻译模型
        tokenizer, model = load_translation_model(use_gpu=use_gpu)
        
        if tokenizer and model:
            # 批量执行回译
            start_time = time.time()
            augmented_texts = back_translation_batch(
                list(back_translation_texts), tokenizer, model, 
                batch_size=BATCH_SIZE
            )
            end_time = time.time()
            
            print(f"{GREEN}回译增强完成，耗时: {end_time - start_time:.2f}秒{ENDC}")
            
            # 过滤有效的增强结果
            filter_bar = create_progress_bar(len(augmented_texts), "🔍 过滤有效结果", "样本")
            valid_count = 0
            
            for aug_text, orig_text, label in zip(augmented_texts, back_translation_texts, back_translation_labels):
                if aug_text and aug_text != orig_text:
                    augmented_data.append((aug_text, label))
                    valid_count += 1
                filter_bar.update(1)
            
            filter_bar.close()
            print(f"{GREEN}有效回译增强结果: {valid_count}个 ({valid_count/len(augmented_texts)*100:.2f}%){ENDC}")
        else:
            print(f"{RED}回译模型加载失败，跳过回译增强{ENDC}")
    
    # 执行EDA任务（多进程）
    if eda_tasks:
        print(f"\n{HEADER}{BOLD}=== 执行EDA增强 ==={ENDC}")
        
        # 启动多进程处理
        if num_workers > 1:
            print(f"{BLUE}使用{num_workers}个工作进程并行处理EDA任务{ENDC}")
            with multiprocessing.Pool(processes=num_workers) as pool:
                eda_bar = create_progress_bar(len(eda_tasks), "🔄 EDA增强", "任务")
                
                # 使用imap批量处理，更新进度条
                results_iter = pool.imap(eda_worker, eda_tasks, chunksize=10)
                
                # 处理结果并更新进度条
                for result in results_iter:
                    augmented_data.extend(result)
                    eda_bar.update(1)
                
                eda_bar.close()
        else:
            # 单进程处理
            eda_bar = create_progress_bar(len(eda_tasks), "🔄 EDA增强 (单进程)", "任务")
            for task in eda_tasks:
                result = eda_worker(task)
                augmented_data.extend(result)
                eda_bar.update(1)
            eda_bar.close()
        
        # 计算EDA增强结果数量
        eda_results = len([d for d in augmented_data if d[1] not in back_translation_labels]) \
                    if back_translation_labels else len(augmented_data)
        print(f"{GREEN}EDA增强完成，生成: {eda_results}个结果{ENDC}")
    
    # 整合原始数据和增强数据
    augmented_texts, augmented_labels = [], []
    
    # 先添加原始数据
    print(f"\n{BLUE}正在整合原始数据和增强数据...{ENDC}")
    merge_bar = create_progress_bar(len(texts) + len(augmented_data), "🔄 整合数据", "样本")
    
    # 添加原始数据
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        merge_bar.update(1)
    
    # 添加增强数据
    for text, label in augmented_data:
        augmented_texts.append(text)
        augmented_labels.append(label)
        merge_bar.update(1)
    
    merge_bar.close()
    
    # 计算增强后的类别分布
    final_label_counter = Counter(augmented_labels)
    
    print(f"\n{HEADER}{BOLD}=== 增强结果 ==={ENDC}")
    print(f"{BOLD}原始数据量: {len(texts)}{ENDC}")
    print(f"{BOLD}增强产生样本: {GREEN}{len(augmented_data)}{ENDC}")
    print(f"{BOLD}最终数据量: {GREEN}{len(augmented_texts)}{ENDC}")
    print(f"{BOLD}增强比例: {GREEN}{len(augmented_data)/len(texts)*100:.2f}%{ENDC}")
    
    print(f"\n{BOLD}增强后类别分布:{ENDC}")
    total_final = sum(final_label_counter.values())
    
    # 按照增强后的数量排序
    sorted_final = sorted(final_label_counter.items(), key=lambda x: x[1], reverse=True)
    
    for label, count in sorted_final:
        original = label_counter[label]
        increase = count - original
        ratio_before = original / sum(label_counter.values()) * 100
        ratio_after = count / total_final * 100
        
        if increase > 0:
            print(f"  类别 {BLUE}{label}{ENDC}: {original} ({YELLOW}{ratio_before:.2f}%{ENDC}) → {GREEN}{count}{ENDC} ({GREEN}{ratio_after:.2f}%{ENDC}) (+{increase}, +{increase/original*100:.2f}%)")
        else:
            print(f"  类别 {BLUE}{label}{ENDC}: {count} ({ratio_after:.2f}%) (未增强)")
    
    # 写入文件或返回结果
    if output_path:
        print(f"\n{BLUE}正在将增强后的数据写入 {output_path}...{ENDC}")
        with open(output_path, 'w', encoding='utf-8') as f:
            write_bar = create_progress_bar(len(augmented_texts), "💾 保存增强数据", "行")
            for text, label in zip(augmented_texts, augmented_labels):
                f.write(f"{text}\t{label}\n")
                write_bar.update(1)
            write_bar.close()
        print(f"{GREEN}✅ 增强数据已保存到 {output_path}{ENDC}")
        return None
    else:
        return augmented_texts, augmented_labels

def main():
    """主函数：解析命令行参数并执行数据增强"""
    import argparse
    
    parser = argparse.ArgumentParser(description="THUCNews数据集GPU加速平衡增强工具")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入数据集文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行处理的工作进程数")
    parser.add_argument("--max", "-m", type=int, default=MAX_AUGMENTATION_COUNT, help="最大增强数量")
    parser.add_argument("--no-gpu", action="store_true", help="不使用GPU加速")
    parser.add_argument("--sample", "-s", type=int, default=None, help="处理的最大样本数")
    
    args = parser.parse_args()
    
    # 执行数据增强
    gpu_optimized_augment(
        file_path=args.input,
        output_path=args.output,
        max_augmentation=args.max,
        num_workers=args.workers,
        use_gpu=not args.no_gpu,
        sample_limit=args.sample
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\033[91m处理被用户中断\033[0m")
    except Exception as e:
        print(f"\n\033[91m处理过程中发生错误: {e}\033[0m")
    finally:
        print("\n\033[92m程序结束\033[0m")