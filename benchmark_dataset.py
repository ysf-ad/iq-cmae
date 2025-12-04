import time
import torch
from torch.utils.data import DataLoader
from iq_cmae.data.ne_data_raw_dataset import NEDataRawDataset
import os

def benchmark():
    print("Initializing Dataset...")
    # Use same settings as training
    ds = NEDataRawDataset(
        data_root='data/ne-data',
        bandwidth='5 GHz Bandwidth',
        image_size=224,
        voltage_split="train",
        subset_ratio=0.01, # Small subset for quick init
        teacher_noise_std=0.0,
        student_noise_std=0.2,
        modality_mask="constellation+gaf+spectrogram",
        cache_dir=None # Disable caching
    )
    
    print(f"Dataset size: {len(ds)}")
    
    # Benchmark __getitem__
    print("Benchmarking __getitem__ (single thread)...")
    start = time.time()
    for i in range(10):
        _ = ds[i]
    end = time.time()
    print(f"__getitem__ avg: {(end - start)/10*1000:.2f} ms")
    
    # Benchmark DataLoader
    print("Benchmarking DataLoader (num_workers=4)...")
    dl = DataLoader(ds, batch_size=16, num_workers=4, shuffle=False)
    
    start = time.time()
    for i, batch in enumerate(dl):
        if i >= 5: break
    end = time.time()
    print(f"DataLoader avg batch (workers=4): {(end - start)/5*1000:.2f} ms")

if __name__ == "__main__":
    # os.environ["CAPC_FORCE_CPU"] = "1" # Test GPU
    benchmark()
