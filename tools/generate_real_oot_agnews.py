#!/usr/bin/env python3
"""
Script to generate LARGE-SCALE, DIVERSE OOT datasets.
Method: Downloads the raw AG News CSV directly from GitHub (Bypassing HF API limits).
Contains 120,000 unique samples to ensure diversity.
"""

import os
import yaml
import logging
import csv
import requests
import io

# ================= 配置区域 =================

DATA_SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/data/oot_generated"
CONFIG_SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/configs/data_conf/Test_Time_Learning/OOT_Generated"
PROJECT_ROOT = "/root/autodl-tmp/MemoryAgentBench"

# 需要生成的 OOT 文件数量
START_INDEX = 7
END_INDEX = 32
TOTAL_FILES_NEEDED = END_INDEX - START_INDEX + 1

# 目标文件大小：每个 OOT 文件大约 1.5MB (约等于原文本量)
# 1.5 MB characters approx matches the token count needed.
TARGET_SIZE_PER_FILE = 1_500_000 

# AG News 原始 CSV 下载链接 (GitHub Raw, 不走 HF API)
CSV_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_parse_csv():
    """下载并解析包含 120,000 条数据的 CSV"""
    logging.info("🚀 Downloading raw CSV data (120k samples)...")
    try:
        response = requests.get(CSV_URL, timeout=60)
        response.raise_for_status()
        
        # 解析 CSV
        # 格式: Class Index, Title, Description
        content = response.content.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content), delimiter=',')
        
        samples = []
        label_map = {
            "1": "World",
            "2": "Sports",
            "3": "Business",
            "4": "Sci/Tech"
        }
        
        for row in csv_reader:
            if len(row) >= 3:
                label_idx = row[0]
                title = row[1]
                desc = row[2]
                
                label_text = label_map.get(label_idx, "News")
                full_text = f"{title}. {desc}"
                
                samples.append({
                    "text": full_text,
                    "label": label_text
                })
                
        logging.info(f"✅ Successfully loaded {len(samples)} unique real samples.")
        return samples
        
    except Exception as e:
        logging.error(f"Failed to download raw CSV: {e}")
        return []

def format_sample(sample):
    """格式化为 MemoryAgentBench 需要的 QA 格式"""
    clean_text = sample['text'].replace("\n", " ").strip()
    return f"Question: {clean_text}\nlabel: {sample['label']}\n\n"

def main():
    # 1. 准备目录
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    
    # 2. 获取海量真实数据
    all_samples = download_and_parse_csv()
    if not all_samples:
        return
    
    total_samples_count = len(all_samples)
    global_sample_ptr = 0 # 全局指针，保证跨文件也不重复（直到用完一轮）

    # 3. 循环生成文件
    for i in range(TOTAL_FILES_NEEDED):
        current_file_id = START_INDEX + i
        file_content = ""
        current_file_size = 0
        
        # 持续填充内容，直到达到目标大小 (1.5MB)
        while current_file_size < TARGET_SIZE_PER_FILE:
            # 线性提取，保证不重复
            sample = all_samples[global_sample_ptr % total_samples_count]
            global_sample_ptr += 1
            
            entry = format_sample(sample)
            file_content += entry
            current_file_size += len(entry)
        
        # 4. 写入 TXT 数据文件
        file_name = f"oot_gen_{current_file_id}.txt"
        file_path = os.path.join(DATA_SAVE_DIR, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
            
        # 5. 写入 YAML 配置文件 (一次性补全所有字段)
        config_name = f"OOT_Gen_{current_file_id}.yaml"
        config_path = os.path.join(CONFIG_SAVE_DIR, config_name)
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        
        yaml_data = {
            "dataset": "Test_Time_Learning",
            "sub_dataset": "ag_news_real_oot",
            "file_path": rel_path,
            "name": f"oot_generated_{current_file_id}",
            "format": "text",
            "seed": 42 + i, # 给每个文件不同的 seed
            "max_test_samples": 0,
            "split": "test",
            "description": f"Real AG News OOT dataset #{current_file_id} (No Repetition)"
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            
        logging.info(f"✅ Generated OOT #{current_file_id}: Size {current_file_size/1024:.2f} KB | Samples from index {global_sample_ptr - (current_file_size // 200)} to {global_sample_ptr}")

if __name__ == "__main__":
    main()