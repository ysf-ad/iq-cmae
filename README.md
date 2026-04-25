# IQ-CMAE: Cross-Modal Masked Autoencoder for RF Sensing

This repository contains the official implementation of **IQ-CMAE**, a self-supervised learning framework for Radio Frequency (RF) sensing that leverages cross-modal masking across Constellation, Gramian Angular Field (GAF), and Spectrogram representations.

In addition to training and linear-probe evaluation, the repo includes integrated encoder-latency measurement and CUDA-graph benchmarking to measure both representation quality and deployment-oriented inference performance from the same codebase.

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
Train a standard MAE-style model with early fusion (Unified Encoder); use MAE directly through this configuration when you need the plain MAE baseline.
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
  --bandwidth "5 GHz Bandwidth"
```

**Key Arguments:**
*   `--checkpoint`: Path to the model checkpoint.
*   `--shots`: Number of samples per class for linear probing (default: 10). Use -1 for full dataset.
*   `--bandwidth`: Optional bandwidth filter for THz evaluation.

### Evaluation With Latency Measurement

`evaluate.py` can also report encoder latency before feature extraction. This is useful when you want accuracy and inference timing from the same entry point.

```bash
python evaluate.py \
  --checkpoint outputs/mid_fusion_optimal/checkpoint-99.pth \
  --shots 10 \
  --bandwidth "5 GHz Bandwidth" \
  --measure_latency \
  --use_cuda_graph \
  --latency_warmup_iters 20 \
  --latency_timing_iters 100 \
  --latency_out outputs/eval_latency.json
```

Additional latency-related arguments:
*   `--measure_latency`: Run an encoder forward-pass latency measurement before linear probing.
*   `--use_cuda_graph`: Also benchmark CUDA graph replay latency on GPU.
*   `--latency_warmup_iters`: Warmup iterations for the latency path.
*   `--latency_timing_iters`: Timed iterations for the latency path.
*   `--latency_out`: Optional JSON output path for the latency results.

The latency JSON includes:
*   eager milliseconds per batch and per sample
*   eager throughput
*   optional CUDA-graph milliseconds per batch and per sample
*   optional CUDA-graph throughput and speedup
*   a max-absolute-difference check between eager and CUDA-graph encoder outputs

### Standalone CUDA-Graph Benchmarks

For more focused latency studies, the repo also includes two standalone benchmark scripts:

1. Encoder-only IQ-CMAE benchmark:
```bash
python iq_cmae/benchmark_iqcmae.py \
  --checkpoint outputs/mid_fusion_optimal/checkpoint-99.pth \
  --device cuda \
  --cuda_graph \
  --output outputs/benchmark_iqcmae.json
```

2. Raw-IQ preprocessing benchmark:
```bash
python benchmark_cuda_graph_preprocess.py \
  --data_root data/ne-data/5\ GHz\ Bandwidth \
  --device cuda \
  --output outputs/benchmark_preprocess.json
```

These scripts are intended for reproducible latency studies and reporting. The integrated `evaluate.py` latency path is the simpler option when you want timing and evaluation from the same workflow.

## 📂 Repository Structure

*   `train.py`: Main training script.
*   `evaluate.py`: Main evaluation script.
*   `benchmark_cuda_graph_preprocess.py`: Raw-IQ preprocessing latency benchmark with CUDA-graph replay.
*   `iq_cmae/`: Source code package.
    *   `models/`: Model definitions (`IQCMAE`).
    *   `data/`: Data loading (`NEDataRawDataset`).
    *   `benchmark_iqcmae.py`: Encoder-only IQ-CMAE latency benchmark.
*   `research_archive/`: Archived research scripts and experiments.

## 📝 Citation
[Citation Placeholder]
