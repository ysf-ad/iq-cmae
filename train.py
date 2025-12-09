import argparse
import os
import math
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

# Imports
from models.iqcmae_model import IQCMAE
from data.ne_data_raw_dataset import NEDataRawDataset
from data.italysig_raw_dataset import ItalySigRawDataset
from training.trainer import IQCMAE_Trainer as Trainer

def get_args_parser():
    parser = argparse.ArgumentParser('IQ-CMAE Training', add_help=False)
    
    # Primary Hyperparameters
    parser.add_argument('--cw', default=2.5, type=float,
                        help='Contrastive weight (lambda)')
    parser.add_argument('--k', default=4, type=int,
                        help='Contrastive last k layers (gradient stop / early exit)')
    parser.add_argument('--s', default=9, type=int,
                        help='Shared layers (0=Unified, >0=Mid-Fusion)')
    
    # Training Config
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--blr', default=1.5e-4, type=float,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--weight_decay', default=0.05, type=float)
    
    # Model Config
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--embed_dim', default=192, type=int,
                        help='Embedding dimension')
    parser.add_argument('--depth', default=12, type=int,
                        help='Encoder depth')
    parser.add_argument('--num_heads', default=3, type=int,
                        help='Number of attention heads')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--fusion_type', default='concat', type=str,
                        help='Fusion type for Mid-Fusion (concat, mean, add)')
    
    # Data Config
    parser.add_argument('--dataset_type', default='ne_data_raw', type=str,
                        help='Dataset type: ne_data_raw or italysig')
    parser.add_argument('--data_path', default='data/ne-data', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./outputs/train_run',
                        help='path where to save, empty for no saving')
    parser.add_argument('--label_depth', default=1, type=int,
                        help='Depth of label folder (1=parent, 2=grandparent)')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # Noise Config
    parser.add_argument('--noise_std', default=0.6, type=float,
                        help='Standard deviation of noise for contrastive pairs')
    parser.add_argument('--teacher_noise_snr_db', default=None, type=float,
                        help='SNR dB for teacher noise (overrides noise_std)')
    parser.add_argument('--student_noise_std', default=0.0, type=float,
                        help='Standard deviation of noise for student')
    parser.add_argument('--student_noise_snr_db', default=None, type=float,
                        help='SNR dB for student noise')
    parser.add_argument('--subset_ratio', default=1.0, type=float,
                        help='Ratio of data to use (for fast debugging/verification)')
    parser.add_argument('--modality_mask', default=None, type=str,
                        help='Modality mask (e.g. "constellation+gaf")')
    parser.add_argument('--cache_dir', default=None, type=str,
                        help='Cache directory')
    
    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configure Datasets
    print("Loading datasets...")
    
    if args.dataset_type == 'italysig':
        # ITALYSIG Dataset
        dataset_train = ItalySigRawDataset(
            data_root=args.data_path,
            image_size=args.input_size,
            teacher_noise_std=args.noise_std,
            teacher_noise_snr_db=args.teacher_noise_snr_db,
            student_noise_std=args.student_noise_std,
            student_noise_snr_db=args.student_noise_snr_db,
            subset_ratio=args.subset_ratio,
            modality_mask="constellation+gaf+spectrogram",
            cache_dir=args.cache_dir,
            seed=args.seed,
            split="train",
            label_depth=args.label_depth
        )
        # For validation, we use the same dataset structure
        dataset_val = ItalySigRawDataset(
            data_root=args.data_path,
            image_size=args.input_size,
            teacher_noise_std=0.0, # Clean for validation
            student_noise_std=0.0,
            subset_ratio=args.subset_ratio,
            modality_mask="constellation+gaf+spectrogram",
            cache_dir=args.cache_dir,
            seed=args.seed, # Same seed for consistent splitting
            split="val",
            label_depth=args.label_depth
        )
        
    else:
        # NE-Data Raw Dataset (Default)
        bandwidths = ['5 GHz Bandwidth', '10 GHz Bandwidth', '20 GHz Bandwidth']
        
        # Training Datasets
        train_datasets = []
        for bw in bandwidths:
            ds = NEDataRawDataset(
                data_root=args.data_path,
                bandwidth=bw,
                image_size=args.input_size,
                voltage_split="train",
                subset_ratio=args.subset_ratio,
                seed=args.seed,
                teacher_noise_std=0.0, # Clean teacher
                student_noise_std=args.noise_std, # Noisy student
                modality_mask="constellation+gaf+spectrogram",
                cache_dir=None # Explicitly disable caching
            )
            train_datasets.append(ds)
        dataset_train = ConcatDataset(train_datasets)
        
        # Validation Datasets (using eval_train split for validation loss)
        val_datasets = []
        for bw in bandwidths:
            ds = NEDataRawDataset(
                data_root=args.data_path,
                bandwidth=bw,
                image_size=args.input_size,
                voltage_split="eval_train", # Use eval_train for validation
                subset_ratio=args.subset_ratio, # Use subset for faster validation
                seed=args.seed,
                teacher_noise_std=0.0,
                student_noise_std=args.noise_std,
                modality_mask="constellation+gaf+spectrogram",
                cache_dir=None # Explicitly disable caching
            )
            val_datasets.append(ds)
        dataset_val = ConcatDataset(val_datasets)

    print(f"Train size: {len(dataset_train)}")
    print(f"Val size: {len(dataset_val)}")

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Model Configuration
    model = IQCMAE(
        img_size=args.input_size,
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
        
        # Hyperparameters
        contrastive_weight=args.cw,
        contrastive_last_k=args.k,
        shared_layers=args.s,
        
        # Options
        fusion_type=args.fusion_type,
        
        # Fixed
        modality_mask="constellation+gaf+spectrogram"
    )
    
    model.to(device)

    # Optimizer
    eff_batch_size = args.batch_size
    lr = args.blr * eff_batch_size / 256
    
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and ('bias' not in n and 'norm' not in n)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and ('bias' in n or 'norm' in n)], 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # Create Output Directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save Config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Start training for {args.epochs} epochs")
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loader_train=data_loader_train,
        data_loader_val=data_loader_val,
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        start_epoch=0,
        writer=None,
        print_freq=20,
        args=args
    )
    
    trainer.train()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
