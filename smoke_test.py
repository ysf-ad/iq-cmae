import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from iq_cmae.models.iqcmae_model import CorrectedProperCMAE

def test_model():
    print("Initializing model...")
    model = CorrectedProperCMAE(img_size=224, patch_size=16, in_chans=6)
    model.eval()
    
    print("Creating dummy input...")
    x = torch.randn(2, 6, 224, 224)
    
    print("Running forward pass...")
    # Forward pass
    loss, _, _, _, _ = model(x)
    print(f"Forward pass successful. Loss: {loss.item()}")

if __name__ == "__main__":
    test_model()
