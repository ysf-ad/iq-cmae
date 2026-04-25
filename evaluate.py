import argparse
import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm

# Add path to import from iq_cmae
sys.path.append('iq_cmae')
from iq_cmae.models.iqcmae_model import IQCMAE as CorrectedProperCMAE
from iq_cmae.data.ne_data_raw_dataset import NEDataRawDataset

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)


def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def measure_encoder_latency(model, sample_batch, device, warmup_iters=20, timing_iters=100, use_cuda_graph=False):
    """
    Measure forward_encoder latency on a fixed sample batch.
    Returns a dict with eager timing, and optionally CUDA-graph timing on GPU.
    """
    model.eval()
    sample_batch = sample_batch.to(device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model.forward_encoder(sample_batch, mask_ratio=0.0)
        _sync_device(device)

        start = time.perf_counter()
        for _ in range(timing_iters):
            _ = model.forward_encoder(sample_batch, mask_ratio=0.0)
        _sync_device(device)
        eager_total = time.perf_counter() - start

    batch_size = int(sample_batch.shape[0])
    eager_ms_per_batch = (eager_total / timing_iters) * 1000.0
    eager_ms_per_sample = eager_ms_per_batch / batch_size
    eager_samples_per_sec = (timing_iters * batch_size) / eager_total if eager_total > 0 else 0.0

    results = {
        "batch_size": batch_size,
        "warmup_iters": int(warmup_iters),
        "timing_iters": int(timing_iters),
        "eager_ms_per_batch": float(eager_ms_per_batch),
        "eager_ms_per_sample": float(eager_ms_per_sample),
        "eager_samples_per_sec": float(eager_samples_per_sec),
        "cuda_graph_enabled": bool(use_cuda_graph and device.type == 'cuda'),
        "cuda_graph_ms_per_batch": None,
        "cuda_graph_ms_per_sample": None,
        "cuda_graph_samples_per_sec": None,
        "cuda_graph_speedup": None,
        "max_abs_diff": None,
    }

    if use_cuda_graph and device.type == 'cuda':
        static_input = sample_batch.clone()
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(warmup_iters):
                graph_out = model.forward_encoder(static_input, mask_ratio=0.0)
        torch.cuda.current_stream().wait_stream(warm_stream)
        _sync_device(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_latent, static_mask, static_restore = model.forward_encoder(static_input, mask_ratio=0.0)

        with torch.no_grad():
            eager_latent, _, _ = model.forward_encoder(sample_batch, mask_ratio=0.0)
        graph.replay()
        _sync_device(device)

        max_abs_diff = float((eager_latent - static_latent).abs().max().item())

        start = time.perf_counter()
        for _ in range(timing_iters):
            graph.replay()
        _sync_device(device)
        graph_total = time.perf_counter() - start

        graph_ms_per_batch = (graph_total / timing_iters) * 1000.0
        graph_ms_per_sample = graph_ms_per_batch / batch_size
        graph_samples_per_sec = (timing_iters * batch_size) / graph_total if graph_total > 0 else 0.0

        results.update({
            "cuda_graph_ms_per_batch": float(graph_ms_per_batch),
            "cuda_graph_ms_per_sample": float(graph_ms_per_sample),
            "cuda_graph_samples_per_sec": float(graph_samples_per_sec),
            "cuda_graph_speedup": float(eager_ms_per_batch / graph_ms_per_batch) if graph_ms_per_batch > 0 else None,
            "max_abs_diff": max_abs_diff,
        })

    return results


def extract_checkpoint_state_dict(checkpoint):
    """Extract the actual model state dict from a checkpoint payload."""
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            return checkpoint['model']
        if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            return checkpoint['model_state_dict']
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint
    raise ValueError("Unsupported checkpoint format: could not find a model state dict")


def load_model_weights_strict(model, checkpoint):
    """
    Load weights only when the checkpoint matches the expected architecture.
    Partial loads silently invalidate evaluation results.
    """
    state_dict = extract_checkpoint_state_dict(checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint is not compatible with CorrectedProperCMAE evaluation. "
            f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

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
                lbls = samples[1] # Assuming tuple (img, label) if not dict
            
            # Forward encoder with mask_ratio=0.0
            # CorrectedProperCMAE.forward_encoder returns (latent, mask, ids_restore)
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
    # We need to infer config from checkpoint or args
    # For now, let's instantiate with defaults and load weights (strict=False)
    # Ideally, we should save args in checkpoint
    # weights_only=False is required because older checkpoints pickle argparse.Namespace
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Try to load config from checkpoint if available
    # If not, use defaults + args overrides
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
        # We might need to pass these as args to evaluate.py if not in checkpoint
        shared_layers=args.s, 
        fusion_type=args.fusion_type,
        contrastive_last_k=args.k
    )
    model.to(device)
    
    # Load Weights
    load_model_weights_strict(model, checkpoint)
    
    # Data Loading
    print("Loading Data...")
    if args.bandwidth:
        bandwidths = [args.bandwidth]
    else:
        bandwidths = ['5 GHz Bandwidth', '10 GHz Bandwidth', '20 GHz Bandwidth']
    
    train_datasets = []
    test_datasets = []
    
    # Use 20% subset for evaluation speed (standard practice in this repo)
    SUBSET_RATIO = 0.2
    
    # Create Global Class Map
    # We need to know all possible classes beforehand to ensure consistency
    # 4 Modulations * 2 Power Levels * 3 Bandwidths = 24 Classes
    # We can hardcode this or discover it. Hardcoding is safer for "publication ready" code.
    
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
            class_map=global_class_map # Enforce global mapping
        )
        test_ds = NEDataRawDataset(
            data_root=args.data_path, bandwidth=bw, image_size=224,
            voltage_split="eval_test", subset_ratio=SUBSET_RATIO, seed=42,
            label_mode="fine_grained",
            class_map=global_class_map # Enforce global mapping
        )
        train_datasets.append(train_ds)
        test_datasets.append(test_ds)
        
    full_train_ds = ConcatDataset(train_datasets)
    full_test_ds = ConcatDataset(test_datasets)
    
    train_loader = DataLoader(full_train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(full_test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if args.measure_latency:
        print("Measuring encoder latency...")
        latency_loader = test_loader
        if len(full_test_ds) == 0 and len(full_train_ds) > 0:
            latency_loader = train_loader
        if len(full_test_ds) == 0 and len(full_train_ds) == 0:
            raise RuntimeError("Latency measurement requested, but both train and test datasets are empty")
        first_test_batch = next(iter(latency_loader))
        if isinstance(first_test_batch, dict):
            latency_imgs = first_test_batch['image']
        else:
            latency_imgs = first_test_batch[0]
        latency_stats = measure_encoder_latency(
            model=model,
            sample_batch=latency_imgs,
            device=device,
            warmup_iters=args.latency_warmup_iters,
            timing_iters=args.latency_timing_iters,
            use_cuda_graph=args.use_cuda_graph,
        )
        latency_stats["checkpoint"] = args.checkpoint
        latency_stats["bandwidth"] = args.bandwidth if args.bandwidth else "all"
        print(json.dumps(latency_stats, indent=2))
        if args.latency_out:
            Path(args.latency_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.latency_out, 'w', encoding='utf-8') as f:
                json.dump(latency_stats, f, indent=2)
    
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
    # Re-compute unique labels after sampling (just in case)
    unique_labels = torch.unique(y_train)
    label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
    
    # Apply mapping
    y_train_mapped = torch.tensor([label_map[y.item()] for y in y_train], device=device)
    y_test_mapped = torch.tensor([label_map[y.item()] for y in y_test], device=device)
    
    # Train Probe
    acc = train_linear_probe(X_train, y_train_mapped, X_test, y_test_mapped, device, len(unique_labels))
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('IQ-CMAE Evaluation')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--data_path', default='data/ne-data', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--shots', default=10, type=int, help='Shots per class (-1 for full)')
    
    # Model Args (Must match training)
    parser.add_argument('--s', default=9, type=int, help='Shared layers')
    parser.add_argument('--k', default=4, type=int, help='Contrastive last k')
    parser.add_argument('--fusion_type', default='concat', type=str)
    parser.add_argument('--embed_dim', default=192, type=int, help='Embedding dimension')
    parser.add_argument('--depth', default=12, type=int, help='Encoder depth')
    parser.add_argument('--num_heads', default=3, type=int, help='Number of attention heads')
    parser.add_argument('--bandwidth', default=None, type=str, help='Specific bandwidth to evaluate (e.g. "5 GHz Bandwidth")')
    parser.add_argument('--measure_latency', action='store_true', help='Measure forward_encoder latency before feature extraction')
    parser.add_argument('--use_cuda_graph', action='store_true', help='Also benchmark CUDA graph replay latency (CUDA only)')
    parser.add_argument('--latency_warmup_iters', default=20, type=int, help='Warmup iterations for latency measurement')
    parser.add_argument('--latency_timing_iters', default=100, type=int, help='Timing iterations for latency measurement')
    parser.add_argument('--latency_out', default='', type=str, help='Optional JSON path for latency results')
    
    args = parser.parse_args()
    main(args)
