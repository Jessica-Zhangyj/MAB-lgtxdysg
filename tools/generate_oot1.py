"""
OOT Noise Generator Script
--------------------------
This script downloads a long public domain novel (Project Gutenberg) and 
slices it into multiple non-overlapping text files. These files are intended 
to be used as "Out-of-Task" (OOT) noise data for evaluating RAG systems.

Usage:
    python generate_oot_data.py
"""

import os
import requests

# === Configuration ===

# Directory where the generated files will be saved
# NOTE: Update this path to match your project structure
SAVE_DIR = "/root/autodl-tmp/MemoryAgentBench/data/oot_books"

# Number of OOT files to generate
NUM_FILES = 7

# Character count per file (Targeting approx. same length as original context)
CHARS_PER_FILE = 220000 

# Source URL: "War and Peace" by Leo Tolstoy (Project Gutenberg)
# Total length approx. 3.2M characters, sufficient for ~14 distinct slices.
SOURCE_URL = "https://www.gutenberg.org/files/2600/2600-0.txt"

def main():
    # Create the directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"🚀 Step 1: Downloading corpus from {SOURCE_URL}...")
    try:
        # Use headers to mimic a browser user-agent to avoid being blocked
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(SOURCE_URL, headers=headers, timeout=60)
        resp.encoding = 'utf-8' # Ensure correct encoding
        full_text = resp.text
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    total_len = len(full_text)
    print(f" Download complete! Total length: {total_len} chars "
          f"(Sufficient for {total_len // CHARS_PER_FILE} slices).")

    print(f" Step 2: Slicing and generating {NUM_FILES} files...")
    
    for i in range(NUM_FILES):
        # Calculate start and end indices to ensure non-overlapping content
        start_idx = i * CHARS_PER_FILE
        end_idx = start_idx + CHARS_PER_FILE
        
        # Handle wrap-around if the source text is exhausted
        if end_idx > total_len:
            print("⚠️ Warning: Source text exhausted. Looping back to the beginning...")
            text_chunk = full_text[start_idx:] + full_text[:end_idx-total_len]
        else:
            text_chunk = full_text[start_idx:end_idx]
            
        # Define output filename and path
        filename = f"oot_noise_{i+1}.txt"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_chunk)
            
        print(f"  - Generated: {filename} (Length: {len(text_chunk)} chars) -> Success")

    print(f"\n All done! Files saved in: {SAVE_DIR}")
    print("The OOT files are now natural text with distinct content, ensuring correct BM25 behavior.")

if __name__ == "__main__":
    main()