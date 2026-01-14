"""
OOT Noise Generator - Dream of the Red Chamber Edition
------------------------------------------------------
Using the English translation of the Chinese classic 'Dream of the Red Chamber' 
(Hung Lou Meng) as OOT noise.

Why this book?
It focuses on domestic life, romance, and poetry in 18th-century China.
The vocabulary (Jade, Matriarch, Garden, Maid) is semantically orthogonal 
to European political history (Treaty, Duke, War), making it perfect for 
testing RAG robustness without keyword collisions.

Source: Project Gutenberg (Trans. H. Bencraft Joly)
Target: 7 distinct files, ~220k chars each.
"""

import os
import requests

# === Configuration ===

SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/data/oot_books"
NUM_FILES = 7
CHARS_PER_FILE = 220000 

# Source URLs: Book 1 and Book 2 of Dream of the Red Chamber
SOURCE_URLS = [
    "https://www.gutenberg.org/files/9603/9603-0.txt", # Book 1
    "https://www.gutenberg.org/files/9604/9604-0.txt"  # Book 2
]

def download_text(url):
    """Helper to download and decode text safely."""
    print(f" Downloading from {url}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.encoding = 'utf-8'
        return resp.text
    except Exception as e:
        print(f" Failed to download {url}: {e}")
        return ""

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Step 1: Download and Combine Sources
    print(f" Step 1: Acquiring 'Dream of the Red Chamber' corpus...")
    full_corpus = ""
    for url in SOURCE_URLS:
        text = download_text(url)
        # Remove Gutenberg headers/footers roughly to keep it clean (optional but good)
        # Simple heuristic: keep the middle 98% to strip legalese
        if len(text) > 10000:
            text = text[5000:-5000] 
        full_corpus += text + "\n\n"

    total_len = len(full_corpus)
    print(f" Corpus ready. Total length: {total_len} chars.")
    
    # Step 2: Slice and Dice
    print(f" Step 2: Generating {NUM_FILES} files of {CHARS_PER_FILE} chars each...")
    
    for i in range(NUM_FILES):
        target_len = CHARS_PER_FILE
        start_idx = (i * target_len) % total_len
        
        # Extract chunk
        chunk = full_corpus[start_idx : start_idx + target_len]
        
        # If chunk is shorter than needed (reached end of book), loop back to start
        if len(chunk) < target_len:
            remainder = target_len - len(chunk)
            chunk += full_corpus[:remainder]
            
        filename = f"oot_noise_{i+1}.txt"
        filepath = os.path.join(SAVE_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)
            
        print(f"  - Generated: {filename} (Length: {len(chunk)}) -> Contains 'Red Chamber' text")

    print(f"\n All done! Saved to: {SAVE_DIR}")
    print("Now your OOT memory is filled with 'Baoyu', 'Daiyu', and 'Grand View Garden'.")
    print("Zero chance of colliding with 'Rollo' or 'Normandy'. Enjoy!")

if __name__ == "__main__":
    main()