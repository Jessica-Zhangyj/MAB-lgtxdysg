#!/usr/bin/env python3
"""
Script to generate "Hard Negative" OOT datasets (Pseudo-QA/Intent format).
Designed to mimic the structure of 'clinic150' and 'trec' to confuse the retriever.
"""

import os
import requests
import yaml
import logging
import random
import re

# ================= 配置区域 =================

# 绝对路径，防止出错
DATA_SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/data/oot_generated"
CONFIG_SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/configs/data_conf/Test_Time_Learning/OOT_Generated"
PROJECT_ROOT = "/root/autodl-tmp/MemoryAgentBench"

# 需要生成的数量 (从 7 到 32)
START_INDEX = 7
END_INDEX = 32
TOTAL_NEEDED = END_INDEX - START_INDEX + 1

# 来源：古腾堡计划小说
BOOK_URLS = [
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
    "https://www.gutenberg.org/files/135/135-0.txt",    # Les Miserables
    "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
]

# 伪装标签 (用来迷惑模型)
# 我们用一些看起来像分类的标签，但其实是随机的
PSEUDO_LABELS = [
    "oot_negative", "irrelevant_class", "noise_data", "unknown_intent", 
    "general_query", "random_talk", "out_of_domain"
]

# 每个 OOT 文件的目标大小 (字符数)
CHARS_PER_FILE = 1_500_000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_corpus(urls):
    full_text = ""
    logging.info("🚀 Starting corpus download...")
    for url in urls:
        try:
            logging.info(f"Downloading: {url}")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                # 简单的清洗，去掉明显的头部版权声明
                text = resp.text
                if len(text) > 10000: text = text[5000:] 
                full_text += text + "\n"
            else:
                logging.warning(f"Failed to download {url}")
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")
    
    if not full_text:
        logging.warning("⚠️ Network failed. Using fallback text.")
        full_text = "This is a fallback text for OOT generation to prevent crash. " * 10000
        
    return full_text

def format_as_pseudo_qa(raw_text_slice):
    """
    核心逻辑：把一段小说文本，强行改成类似 Clinic150 的格式。
    
    Clinic150 格式通常是:
    [用户的话]
    label: [意图分类]
    """
    formatted_content = ""
    
    # 1. 按行或句子分割
    # 这里我们简单按换行符分割，并过滤掉太短的行
    lines = raw_text_slice.split('\n')
    
    buffer = ""
    for line in lines:
        line = line.strip()
        if len(line) < 20: continue # 跳过空行和短句
        
        # 2. 构造伪装格式
        # 我们可以模仿 main.py 中看到的格式: "Question: ... label: ..."
        # 也可以模仿 Clinic 的格式。为了通用性，我们用 Question/Label 结构。
        
        fake_label = random.choice(PSEUDO_LABELS)
        
        # 关键：这里模拟了你要跑的数据集的格式！
        entry = f"Question: {line}\nlabel: {fake_label}\n\n"
        
        formatted_content += entry
        
    return formatted_content

def main():
    # 1. 准备目录
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    
    # 2. 下载语料
    corpus = download_corpus(BOOK_URLS)
    total_len = len(corpus)
    logging.info(f"Corpus ready: {total_len} chars")

    # 3. 循环生成
    for i in range(TOTAL_NEEDED):
        current_id = START_INDEX + i
        
        # 循环切片
        start = (i * CHARS_PER_FILE) % total_len
        end = start + CHARS_PER_FILE
        if end > total_len:
            raw_slice = corpus[start:] + corpus[:end-total_len]
        else:
            raw_slice = corpus[start:end]
            
        # 🔥 核心：转换为 QA 格式
        qa_content = format_as_pseudo_qa(raw_slice)
        
        # 写入 TXT
        file_name = f"oot_gen_{current_id}.txt"
        file_path = os.path.join(DATA_SAVE_DIR, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(qa_content)
            
        # 写入 YAML (一次性补全所有字段)
        config_name = f"OOT_Gen_{current_id}.yaml"
        config_path = os.path.join(CONFIG_SAVE_DIR, config_name)
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        
        yaml_data = {
            "dataset": "Test_Time_Learning",
            "sub_dataset": "generated_oot_slice",
            "file_path": rel_path,
            "name": f"oot_generated_{current_id}",
            "format": "text",
            "seed": 42,
            "max_test_samples": 0,
            "split": "test",
            "description": f"Hard Negative OOT dataset #{current_id} (Pseudo-QA format)"
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            
        logging.info(f"✅ Generated OOT #{current_id} (Pseudo-QA format)")

if __name__ == "__main__":
    main()
    