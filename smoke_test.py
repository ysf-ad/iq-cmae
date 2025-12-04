import torch
import sys
import os

# Add project root to path (current directory)
sys.path.append(os.getcwd())

from models.iqcmae_model import IQCMAE

def test_model():
    print("Initializing model...")
    model = IQCMAE(img_size=224, patch_size=16, in_chans=6)
    model.eval()
    
    x = torch.randn(2, 6, 224, 224)
    
    print("Running forward pass...")
    # Forward pass
    loss, _, _, _, _ = model(x)
    print(f"Forward pass successful. Loss: {loss.item()}")

if __name__ == "__main__":
    test_model()
