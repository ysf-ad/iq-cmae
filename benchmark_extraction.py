import time
import numpy as np
from iq_cmae.utils.iq_extractor import extract_iq_data
import os
import glob

def benchmark():
    # Find a sample file
    files = glob.glob("data/ne-data/**/*.sigmf-data", recursive=True)
    if not files:
        print("No files found!")
        return
    
    file_path = files[0]
    meta_path = file_path.replace('.sigmf-data', '.sigmf-meta')
    
    print(f"Benchmarking extraction on {file_path}")
    
    # Warmup
    extract_iq_data(file_path, meta_path)
    
    start = time.time()
    for _ in range(10):
        extract_iq_data(file_path, meta_path)
    end = time.time()
    
    print(f"Extraction avg: {(end - start)/10*1000:.2f} ms")

if __name__ == "__main__":
    benchmark()
