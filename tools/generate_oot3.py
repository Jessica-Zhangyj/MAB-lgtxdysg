"""
OOT Library Generator
---------------------
Downloads multiple diverse, non-historical public domain books to create 
a massive corpus (approx 8M+ chars), then slices it into 32 distinct 
files of 220,000 characters each.

Selected Domains:
- Whaling & Ocean (Moby Dick)
- Victorian Domestic Life (David Copperfield)
- Mystery & Crime (Sherlock Holmes)
- Adventure & Revenge (Count of Monte Cristo)

These domains are semantically orthogonal to "European Political History" (Treaties/Wars).
"""

import os
import requests

# === Configuration ===

SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/data/oot_books"
NUM_FILES_NEEDED = 32
CHARS_PER_FILE = 220000 

# Source URLs from Project Gutenberg
BOOKS = {
    "Moby Dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "David Copperfield": "https://www.gutenberg.org/files/766/766-0.txt",
    "Sherlock Holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "Count of Monte Cristo": "https://www.gutenberg.org/files/1184/1184-0.txt"
}

def download_book(title, url):
    """Download a single book with error handling."""
    print(f"📥 Downloading '{title}'...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.encoding = 'utf-8'
        text = resp.text
        
        # Simple cleanup: Remove the first and last 2% to strip Project Gutenberg headers
        # This prevents the legal license text from repeating in your dataset
        trim_size = int(len(text) * 0.02)
        return text[trim_size : -trim_size]
        
    except Exception as e:
        print(f"❌ Failed to download {title}: {e}")
        return ""

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Step 1: Build the Super Corpus
    print(f"🚀 Step 1: Building the Super Corpus from {len(BOOKS)} books...")
    full_corpus = ""
    
    for title, url in BOOKS.items():
        text = download_book(title, url)
        print(f"   + Added '{title}' ({len(text)} chars)")
        full_corpus += text + "\n\n" * 10  # Add spacing between books

    total_len = len(full_corpus)
    required_len = NUM_FILES_NEEDED * CHARS_PER_FILE
    
    print(f"\n✅ Corpus Ready! Total Length: {total_len} chars")
    print(f"📊 Requirement: {NUM_FILES_NEEDED} files * {CHARS_PER_FILE} = {required_len} chars")
    
    if total_len < required_len:
        print("⚠️  Warning: Total text is slightly shorter than required. Some content will loop.")
    else:
        print("✨ Perfect! We have enough unique text for non-overlapping files.")

    # Step 2: Slice into 32 Files
    print(f"\n🚀 Step 2: Slicing into {NUM_FILES_NEEDED} files...")
    
    for i in range(NUM_FILES_NEEDED):
        # Calculate start index
        start_idx = (i * CHARS_PER_FILE) % total_len
        end_idx = start_idx + CHARS_PER_FILE
        
        # Extract chunk
        if end_idx <= total_len:
            chunk = full_corpus[start_idx : end_idx]
        else:
            # Handle wrap-around if we run out of text (unlikely with this list)
            chunk = full_corpus[start_idx:] + full_corpus[:end_idx-total_len]
            
        filename = f"oot_noise_{i+1}.txt"
        filepath = os.path.join(SAVE_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)
            
        print(f"  - Generated: {filename} (220k chars) -> Success")

    print(f"\n🎉 All 32 files are ready in: {SAVE_DIR}")
    print("These files contain a mix of whales, detectives, and Victorian families.")
    print("Your BM25 retriever will find them very easy to distinguish from your Original task!")

if __name__ == "__main__":
    main()