
#!/usr/bin/env python3
"""
Script to generate supplementary "Out-of-Task" (OOT) datasets for RAG benchmarking.

This script downloads public domain literature (Project Gutenberg), slices it into 
fixed-size chunks, and generates corresponding YAML configuration files. This is 
used to fill the quota of required OOT documents when existing datasets are insufficient.

Author: MemoryAgentBench Contributor
License: MIT
"""

import os
import requests
import yaml
import logging

# ================= CONFIGURATION =================

# Paths for saving generated data and configs
# Ensure these paths align with your project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SAVE_DIR = os.path.join(BASE_DIR, "data", "oot_generated")
CONFIG_SAVE_DIR = os.path.join(BASE_DIR, "configs", "data_conf", "Test_Time_Learning", "OOT_Generated")

# Dataset Parameters
TOTAL_OOT_REQUIRED = 32
EXISTING_DATASETS_COUNT = 6  # You already have 5 ICL + 1 RecSys
START_INDEX = EXISTING_DATASETS_COUNT + 1  # Start numbering from 7

# Content Generation
# Target length per document (chars). Set to match your RecSys/Banking77 length.
# 1.5M chars is approx. 300k-400k tokens.
CHARS_PER_DOCUMENT = 1_480_000 

# Source Corpus (Public Domain Books from Project Gutenberg)
# Using varied genres (Adventure, Sci-Fi, Romance) to provide diverse noise.
BOOK_URLS = [
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick (Melville)
    "https://www.gutenberg.org/files/135/135-0.txt",    # Les Miserables (Hugo)
    "https://www.gutenberg.org/files/1184/1184-0.txt",   # Count of Monte Cristo (Dumas)
    "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein (Shelley)
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice (Austen)
]

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_clean_text(urls):
    """
    Downloads text from URLs and removes header/footer noise.
    
    Args:
        urls (list): List of string URLs to text files.
        
    Returns:
        str: A single concatenated string of the corpus.
    """
    full_text = ""
    logging.info("Starting corpus download...")

    for url in urls:
        try:
            logging.info(f"Downloading: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Simple heuristic to strip Project Gutenberg headers/footers
            # Usually the first and last 5000 characters contain license info
            raw_text = response.text
            if len(raw_text) > 10000:
                clean_content = raw_text[5000:-5000]
            else:
                clean_content = raw_text
                
            full_text += clean_content + "\n\n"
            
        except requests.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")

    logging.info(f"Corpus prepared. Total length: {len(full_text):,} characters.")
    return full_text

def generate_datasets(full_text):
    """
    Slices the corpus into chunks and writes data/config files.
    
    Args:
        full_text (str): The complete source text.
    """
    if not full_text:
        logging.error("No text available to generate datasets. Exiting.")
        return

    generated_configs = []
    total_len = len(full_text)
    
    # Calculate how many datasets we need to generate
    needed_count = TOTAL_OOT_REQUIRED - EXISTING_DATASETS_COUNT
    
    logging.info(f"Generating {needed_count} datasets (OOT {START_INDEX} to {TOTAL_OOT_REQUIRED})...")

    for i in range(needed_count):
        current_id = START_INDEX + i
        
        # Calculate slice indices with wrapping (looping) logic
        # This ensures we always have full-length documents even if corpus is small
        start_idx = (i * CHARS_PER_DOCUMENT) % total_len
        end_idx = start_idx + CHARS_PER_DOCUMENT
        
        if end_idx > total_len:
            # Wrap around: take the end of the text and append the beginning
            remainder = end_idx - total_len
            doc_content = full_text[start_idx:] + full_text[:remainder]
        else:
            doc_content = full_text[start_idx:end_idx]

        # 1. Write the content to a .txt file
        file_name = f"oot_gen_{current_id}.txt"
        file_path = os.path.join(DATA_SAVE_DIR, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
        except IOError as e:
            logging.error(f"Failed to write data file {file_path}: {e}")
            continue

        # 2. Generate the YAML configuration
        config_name = f"OOT_Gen_{current_id}.yaml"
        config_path = os.path.join(CONFIG_SAVE_DIR, config_name)
        
        yaml_data = {
            "dataset": "Test_Time_Learning",
            "sub_dataset": "generated_oot_slice",
            "file_path": file_path,
            "name": f"oot_generated_{current_id}",
            "format": "text",
            "description": f"Auto-generated OOT noise dataset #{current_id} from public domain literature."
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            generated_configs.append(config_path)
            logging.info(f"Generated OOT #{current_id}: {config_name}")
            
        except IOError as e:
            logging.error(f"Failed to write config file {config_path}: {e}")

    return generated_configs

def main():
    # Ensure output directories exist
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)

    # 1. Prepare Data
    corpus = download_and_clean_text(BOOK_URLS)
    
    # 2. Generate Files
    configs = generate_datasets(corpus)

    # 3. Output Instructions
    if configs:
        print("\n" + "="*60)
        print("SUCCESS! Generated the following configuration files.")
        print("Copy and paste the list below into your main YAML config")
        print("under 'external_dataset_configs':")
        print("="*60 + "\n")
        
        for config in configs:
            # Output relative path for cleaner YAML
            rel_path = os.path.relpath(config, BASE_DIR)
            print(f"    - {rel_path}")
        print("\n" + "="*60)

if __name__ == "__main__":
    main()