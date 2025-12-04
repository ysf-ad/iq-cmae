import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm

# Imports
from models.iqcmae_model import IQCMAE
from data.ne_data_raw_dataset import NEDataRawDataset
from data.italysig_raw_dataset import ItalySigRawDataset

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for samples in tqdm(dataloader, desc="Extracting"):
            if isinstance(samples, dict):
                imgs = samples['image'].to(device)
                lbls = samples['label']
            else:
                imgs = samples[0].to(device)
                lbls = samples[1]
            
            # Forward encoder with mask_ratio=0.0
            # IQCMAE.forward_encoder returns (latent, mask, ids_restore)
            # We need the CLS token or mean pool
            latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
            
            # Use CLS token (index 0)
            cls_token = latent[:, 0]
            
            features.append(cls_token.cpu())
            labels.append(lbls.cpu())
            
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_linear_probe(X_train, y_train, X_test, y_test, device, num_classes, epochs=100):
    print(f"Training Linear Probe on {len(X_train)} samples...")
    
    # Standardize features
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    input_dim = X_train.shape[1]
    probe = LinearProbe(input_dim, num_classes).to(device)
    
    # AdamW with Cosine Schedule (Standard for LP)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    probe.train()
    for epoch in range(epochs):
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = probe(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    # Evaluate
    probe.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    with torch.no_grad():
        output = probe(X_test)
        pred = output.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
        
    return acc

def main(args):
    device = torch.device(args.device)
    print(f"Evaluating checkpoint: {args.checkpoint}")
    
    # Load Model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = CorrectedProperCMAE(
        img_size=224,
        patch_size=16,
        in_chans=6,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        
        # Important: These must match training for correct architecture
        shared_layers=args.s, 
        fusion_type=args.fusion_type,
        contrastive_last_k=args.k,
    )
    model.to(device)
    
    # Load Weights
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    # Data Loading
    print("Loading Data...")
    
    train_datasets = []
    test_datasets = []
    
    # Use 20% subset for evaluation speed (standard practice in this repo)
    SUBSET_RATIO = args.subset_ratio
    
    if args.dataset_type == 'italysig':
        # ITALYSIG
        train_ds = ItalySigRawDataset(
            data_root=args.data_path, image_size=224,
            subset_ratio=SUBSET_RATIO, seed=42,
            modality_mask="constellation+gaf+spectrogram"
        )
        test_ds = ItalySigRawDataset(
            data_root=args.data_path, image_size=224,
            subset_ratio=SUBSET_RATIO, seed=43,
            modality_mask="constellation+gaf+spectrogram"
        )
        train_datasets.append(train_ds)
        test_datasets.append(test_ds)
        
    else:
        # NE-DATA
        if args.bandwidth:
            bandwidths = [args.bandwidth]
        else:
            bandwidths = ['5 GHz Bandwidth', '10 GHz Bandwidth', '20 GHz Bandwidth']
        
        # Create Global Class Map for NE-Data
        MODULATIONS = ["16QAM", "4PSK", "64QAM", "8PSK"]
        POWER_LEVELS = ["600mV", "75mV"]
        BANDWIDTHS = ["5 GHz Bandwidth", "10 GHz Bandwidth", "20 GHz Bandwidth"]
        
        global_class_map = {}
        idx = 0
        for bw in sorted(BANDWIDTHS):
            bw_clean = bw.replace(" ", "")
            for mod in sorted(MODULATIONS):
                for pwr in sorted(POWER_LEVELS):
                    class_name = f"{bw_clean}_{mod}_{pwr}"
                    global_class_map[class_name] = idx
                    idx += 1
                    
        print(f"Global Class Map: {len(global_class_map)} classes")
        
        for bw in bandwidths:
            train_ds = NEDataRawDataset(
                data_root=args.data_path, bandwidth=bw, image_size=224,
                voltage_split="eval_train", subset_ratio=SUBSET_RATIO, seed=42,
                label_mode="fine_grained",
                class_map=global_class_map
            )
            test_ds = NEDataRawDataset(
                data_root=args.data_path, bandwidth=bw, image_size=224,
                voltage_split="eval_test", subset_ratio=SUBSET_RATIO, seed=42,
                label_mode="fine_grained",
                class_map=global_class_map
            )
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
        
    full_train_ds = ConcatDataset(train_datasets)
    full_test_ds = ConcatDataset(test_datasets)
    
    train_loader = DataLoader(full_train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(full_test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Extract Features
    print("Extracting Features...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)
    
    unique_labels = torch.unique(y_train)
    num_classes = len(unique_labels)
    print(f"Classes: {num_classes}")
    
    # Few-shot Sampling
    if args.shots > 0:
        print(f"Sampling {args.shots}-shot subset...")
        indices = []
        for c in unique_labels:
            c_indices = (y_train == c).nonzero(as_tuple=True)[0]
            if len(c_indices) >= args.shots:
                indices.append(c_indices[torch.randperm(len(c_indices))[:args.shots]])
            else:
                indices.append(c_indices)
        indices = torch.cat(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"New train size: {len(X_train)}")
        
    # Remap labels to 0..N-1
    unique_labels = torch.unique(y_train)
    label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
    
    y_train_mapped = torch.tensor([label_map[y.item()] for y in y_train], device=device)
    y_test_mapped = torch.tensor([label_map[y.item()] for y in y_test], device=device)
    
    # Train Probe
    acc = train_linear_probe(X_train, y_train_mapped, X_test, y_test_mapped, device, len(unique_labels))
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('IQ-CMAE Linear Probe Evaluation')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--dataset_type', default='ne_data_raw', type=str)
    parser.add_argument('--data_path', default='data/ne-data', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--shots', default=10, type=int, help='Shots per class (-1 for full)')
    parser.add_argument('--subset_ratio', default=0.2, type=float, help='Subset ratio for feature extraction')
    
    # Model Args (Must match training)
    parser.add_argument('--s', default=9, type=int, help='Shared layers')
    parser.add_argument('--k', default=4, type=int, help='Contrastive last k')
    parser.add_argument('--fusion_type', default='concat', type=str)
    
    parser.add_argument('--embed_dim', default=192, type=int, help='Embedding dimension')
    parser.add_argument('--depth', default=12, type=int, help='Encoder depth')
    parser.add_argument('--num_heads', default=3, type=int, help='Number of attention heads')
    parser.add_argument('--bandwidth', default=None, type=str, help='Specific bandwidth to evaluate (e.g. "5 GHz Bandwidth")')
    
    args = parser.parse_args()
    main(args)
