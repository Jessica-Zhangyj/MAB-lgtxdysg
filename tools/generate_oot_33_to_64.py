#!/usr/bin/env python3
"""
Script to generate OOT datasets #33-#64 using a MIX of AG News and Banking77.
Since Yahoo Answers URLs are unstable, we combine our two known-good sources
(130k+ total samples) and shuffle them differently to create diverse hard negatives.
"""

import os
import yaml
import logging
import csv
import requests
import io
import random

# ================= 配置区域 =================

DATA_SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/data/oot_generated"
CONFIG_SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/configs/data_conf/Test_Time_Learning/OOT_Generated"
PROJECT_ROOT = "/root/autodl-tmp/MemoryAgentBench"

# 生成范围：33 到 64
START_INDEX = 33
END_INDEX = 64
TOTAL_FILES_NEEDED = END_INDEX - START_INDEX + 1
TARGET_SIZE_PER_FILE = 1_500_000 

# 已知 100% 可用的下载链接
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_ag_news():
    logging.info("📥 Downloading AG News...")
    try:
        response = requests.get(AG_NEWS_URL, timeout=60)
        response.raise_for_status()
        content = response.content.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content), delimiter=',')
        samples = []
        for row in csv_reader:
            if len(row) >= 3:
                # AG News: Title + Description
                text = row[1] + ". " + row[2]
                samples.append({"text": text, "label": "News_Topic"})
        return samples
    except Exception as e:
        logging.error(f"AG News download failed: {e}")
        return []

def download_banking77():
    logging.info("📥 Downloading Banking77...")
    try:
        response = requests.get(BANKING77_URL, timeout=60)
        response.raise_for_status()
        content = response.content.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content), delimiter=',')
        next(csv_reader) # Skip header
        samples = []
        for row in csv_reader:
            if len(row) >= 2:
                samples.append({"text": row[0], "label": row[1]})
        return samples
    except Exception as e:
        logging.error(f"Banking77 download failed: {e}")
        return []

def format_sample(sample):
    clean_text = sample['text'].replace("\n", " ").strip()
    return f"Question: {clean_text}\nlabel: {sample['label']}\n\n"

def main():
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    
    # 1. 下载所有数据
    ag_data = download_ag_news()
    bank_data = download_banking77()
    
    if not ag_data and not bank_data:
        logging.error("❌ Both downloads failed. Check network.")
        return

    # 2. 合并数据池
    all_samples = ag_data + bank_data
    logging.info(f"✅ Total merged pool: {len(all_samples)} samples.")
    
    # 3. 关键：使用新的随机种子打乱，确保和之前生成的顺序不同
    # 这样生成的 33-64 号文件内容也是全新的组合
    random.seed(999) 
    random.shuffle(all_samples)
    
    total_samples = len(all_samples)
    global_ptr = 0

    print("="*60)
    print(f"Generating Mixed Hard-Negative OOTs {START_INDEX} to {END_INDEX}...")
    print("="*60)

    for i in range(TOTAL_FILES_NEEDED):
        current_file_id = START_INDEX + i
        file_content = ""
        current_size = 0
        
        # 填充文件
        while current_size < TARGET_SIZE_PER_FILE:
            sample = all_samples[global_ptr % total_samples]
            global_ptr += 1
            
            entry = format_sample(sample)
            file_content += entry
            current_size += len(entry)
        
        # 写入 TXT
        file_name = f"oot_gen_{current_file_id}.txt"
        file_path = os.path.join(DATA_SAVE_DIR, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
            
        # 写入 YAML
        config_name = f"OOT_Gen_{current_file_id}.yaml"
        config_path = os.path.join(CONFIG_SAVE_DIR, config_name)
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        
        yaml_data = {
            "dataset": "Test_Time_Learning",
            "sub_dataset": "mixed_hard_oot", # 标记为混合数据集
            "file_path": rel_path,
            "name": f"oot_generated_{current_file_id}",
            "format": "text",
            "seed": 100 + i,
            "max_test_samples": 0,
            "split": "test",
            "description": f"Mixed Hard Negative OOT #{current_file_id}"
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            
        logging.info(f"Generated OOT #{current_file_id} (Size: {current_size/1024/1024:.2f} MB)")

    print("\n✅ Success! New OOTs 33-64 generated.")

if __name__ == "__main__":
    main()