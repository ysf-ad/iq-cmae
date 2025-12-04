# IQ-CMAE: Cross-Modal Masked Autoencoder for RF Sensing

This repository contains the official implementation of **IQ-CMAE**, a self-supervised learning framework for Radio Frequency (RF) sensing that leverages cross-modal masking across Constellation, Gramian Angular Field (GAF), and Spectrogram representations.

## 🚀 Quick Start

### Prerequisites
*   Python 3.8+
*   PyTorch 2.0+
*   CUDA-capable GPU recommended

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation
Ensure your dataset is located in `ne-data/` or specify the path using the `--data_root` argument.

## 🏃 Training

The `train.py` script serves as the unified entry point for training both the Unified Baseline and Mid-Fusion models.

### 1. Unified Baseline (CW=0)
Train a standard MAE-style model with early fusion (Unified Encoder).
```bash
python train.py \
  --cw 0 --s 0 --k 0 \
  --output_dir outputs/unified_baseline \
  --epochs 100 --batch_size 64
```

### 2. Mid-Fusion Optimal (CW=2.5, S=9, K=4)
Train the optimal IQ-CMAE model with mid-fusion, contrastive learning, and gradient stopping.
```bash
python train.py \
  --cw 2.5 --s 9 --k 4 \
  --output_dir outputs/mid_fusion_optimal \
  --epochs 100 --batch_size 64
```

**Key Arguments:**
*   `--cw`: Contrastive Weight (default: 2.5)
*   `--s`: Shared Layers (default: 9). Set to 0 for Unified.
*   `--k`: Contrastive Gradient Stopping Layers (default: 4).
*   `--subset_ratio`: Use a fraction of the data (e.g., 0.1) for fast debugging.

## 📊 Evaluation

The `evaluate.py` script performs Linear Probing evaluation on trained checkpoints.

```bash
python evaluate.py \
  --checkpoint outputs/mid_fusion_optimal/checkpoint-99.pth \
  --shots 10 \
  --batch_size 64
```

**Key Arguments:**
*   `--checkpoint`: Path to the model checkpoint.
*   `--shots`: Number of samples per class for linear probing (default: 10). Use -1 for full dataset.

## 📂 Repository Structure

*   `train.py`: Main training script.
*   `evaluate.py`: Main evaluation script.
*   `iq_cmae/`: Source code package.
    *   `models/`: Model definitions (`CorrectedProperCMAE`).
    *   `data/`: Data loading (`NEDataRawDataset`).
*   `research_archive/`: Archived research scripts and experiments.

## 📝 Citation
[Citation Placeholder]
