import time
import numpy as np
import os
import glob
from iq_cmae.data.transforms import create_constellation_image, create_gaf_image, create_spectrogram_custom, create_iq_square_image

def benchmark():
    # Generate random I/Q data (2, 10000)
    iq_data = np.random.randn(2, 10000).astype(np.float32)
    
    # Params
    image_size = 224
    spec_params = {'nperseg': 192, 'noverlap': 144, 'nfft': 384}
    
    print("Benchmarking transforms...")
    
    # Benchmark Constellation
    start = time.time()
    for _ in range(10):
        create_constellation_image(iq_data, image_size)
    end = time.time()
    print(f"Constellation: {(end - start)/10*1000:.2f} ms")
    
    # Benchmark GAF
    start = time.time()
    for _ in range(10):
        create_gaf_image(iq_data, image_size)
    end = time.time()
    print(f"GAF: {(end - start)/10*1000:.2f} ms")
    
    # Benchmark Spectrogram
    start = time.time()
    for _ in range(10):
        create_spectrogram_custom(iq_data, spec_params, image_size)
    end = time.time()
    print(f"Spectrogram: {(end - start)/10*1000:.2f} ms")
    
    # Benchmark IQ Square
    start = time.time()
    for _ in range(10):
        create_iq_square_image(iq_data, image_size)
    end = time.time()
    print(f"IQ Square: {(end - start)/10*1000:.2f} ms")

if __name__ == "__main__":
    # Force CPU for fair comparison with workers
    os.environ["CAPC_FORCE_CPU"] = "1"
    benchmark()
