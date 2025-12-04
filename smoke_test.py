import sys
import os

# Add project root to path (current directory)
sys.path.append(os.getcwd())
import torch
import os
from models.iqcmae_model import IQCMAE
from data.ne_data_raw_dataset import NEDataRawDataset
from data.italysig_raw_dataset import ItalySigRawDataset

def test_model_forward():
    print("Initializing model...")
    model = IQCMAE(
        img_size=224,
        patch_size=16,
        in_chans=6,
        embed_dim=192,
        depth=4,
        num_heads=3,
        decoder_embed_dim=192,
        decoder_depth=2,
        decoder_num_heads=3,
        mlp_ratio=4.,
        contrastive_weight=1.0,
        contrastive_last_k=1,
        shared_layers=2,
        fusion_type='concat',
        modality_mask="constellation+gaf+spectrogram"
    )

    print("Creating dummy input...")
    x = torch.randn(2, 6, 224, 224)

    print("Running forward pass...")
    loss, loss_recon, loss_contrastive, _, _ = model(x)
    print(f"Forward pass successful. Total Loss: {loss.item()}, Recon: {loss_recon.item()}, Contrastive: {loss_contrastive.item()}")

def test_italysig_dataset():
    print("\nTesting ItalySigRawDataset...")
    data_path = "data/italysig"
    if not os.path.exists(data_path):
        print(f"Path {data_path} not found. Skipping.")
        return

    try:
        ds = ItalySigRawDataset(
            data_root=data_path,
            image_size=224,
            subset_ratio=0.1, # Small subset
            modality_mask="constellation+gaf+spectrogram",
            split="train"
        )
        print(f"Dataset initialized. Size: {len(ds)}")
        if len(ds) > 0:
            item = ds[0]
            print(f"Item 0 loaded. Shape: {item['image'].shape}")
        else:
            print("No samples found in ItalySig.")
    except Exception as e:
        print(f"ItalySig test failed: {e}")

def test_ne_data_dataset():
    print("\nTesting NEDataRawDataset...")
    # Check possible paths
    possible_paths = ["data/ne-data", "data/IQ_recordings"]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
    
    if not data_path:
        print("No NE-Data path found. Skipping.")
        return

    print(f"Using data path: {data_path}")
    try:
        # Try with a common bandwidth folder if structure matches
        ds = NEDataRawDataset(
            data_root=data_path,
            bandwidth="5 GHz Bandwidth", # Assumption
            image_size=224,
            voltage_split="train",
            subset_ratio=0.1,
            modality_mask="constellation+gaf+spectrogram"
        )
        print(f"Dataset initialized. Size: {len(ds)}")
        if len(ds) > 0:
            item = ds[0]
            print(f"Item 0 loaded. Shape: {item['image'].shape}")
        else:
            print("No samples found in NE-Data (or wrong structure).")
    except Exception as e:
        print(f"NE-Data test failed: {e}")

if __name__ == "__main__":
    test_model_forward()
    test_italysig_dataset()
    test_ne_data_dataset()
