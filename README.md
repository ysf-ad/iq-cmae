# IQ-CMAE: Multi-modal Contrastive Masked Autoencoder

Official implementation of **IQ-CMAE**, a multi-modal self-supervised learning framework for wireless signal classification (Bandwidth, Power, Modulation) using raw IQ traces.

## Key Features

*   **Multi-modal Fusion**: Integrates Spectrograms, GAF, and Constellations into a unified representation.
*   **Fairly Parameterized Fusion**: Uses a fusion depth parameter $S$ to balance modality-specific vs. shared layers while maintaining constant total parameters.
*   **Hybrid Objective**: Jointly optimizes Masked Autoencoder (MAE) reconstruction and Contrastive Learning (InfoNCE).
*   **Contrastive Gradient Stopping**: Restricts contrastive gradients to the top $K$ layers ("Last-K"), preventing interference with low-level reconstruction.

## Project Structure

```
iq_cmae/
├── data/                    # Data loading and processing
│   ├── ne_data_raw_dataset.py  # Raw SIGMF dataset loader
│   ├── italysig_raw_dataset.py # ITALYSIG dataset loader
│   ├── transforms.py           # Signal transforms
│   └── ...
├── models/                  # Model architectures
│   ├── iqcmae_model.py         # Main IQ-CMAE model
│   ├── mae_backbone.py         # Base MAE components
│   └── modules.py              # Transformer blocks
├── training/                # Training utilities
│   └── trainer.py              # Trainer class
├── train.py                 # Main training script
└── linear_probe.py          # Evaluation script
```

## Usage

### Training (Pre-training)

Run `train.py` to pre-train the model using on-the-fly data generation:

```bash
python train.py \
    --dataset_type ne_data_raw \
    --data_path /path/to/dataset \
    --epochs 100 \
    --batch_size 64 \
    --contrastive_weight 2.5 \
    --noise_std 0.6 \
    --contrastive_last_k 4 \
    --fusion_type concat \
    --shared_layers 9
```

**Arguments:**
*   `--contrastive_weight`: Weight for contrastive loss (default: `2.5`).
*   `--noise_std`: Teacher branch noise standard deviation (default: `0.6`).
*   `--contrastive_last_k`: Top layers updated by contrastive loss (default: `4`).
*   `--shared_layers`: Number of shared encoder layers (default: `9`).

### Linear Probing / Evaluation

Use the `linear_probe.py` script (located in the project root) to evaluate a pre-trained checkpoint:

```bash
python linear_probe.py \
    --checkpoint outputs/train_run/checkpoint-99.pth \
    --dataset_type ne_data_raw \
    --data_path /path/to/dataset \
    --shots 10 \
    --subset_ratio 0.2 \
    --device cuda
```

**Arguments:**
*   `--checkpoint`: Path to the pre-trained model checkpoint.
*   `--shots`: Number of samples per class for few-shot evaluation (use `-1` for full dataset).
*   `--subset_ratio`: Ratio of data to use for feature extraction (default: `0.2`).
*   `--dataset_type`: `ne_data_raw` or `italysig`.

The script will:
1.  Load the pre-trained encoder.
2.  Extract features (CLS tokens) from the dataset.
3.  Train a linear classifier on the frozen features.
4.  Report test accuracy.

## Architecture Details

**Input**: $224 \times 224 \times 6$ image split into:
1.  **Constellation** (3 ch)
2.  **GAF** (2 ch)
3.  **Spectrogram** (1 ch)

**Forward Pass**:
1.  **Split & Embed**: Inputs are split and processed by separate `PatchEmbed` layers.
2.  **Private Encode**: Modalities pass through independent Transformer blocks.
3.  **Fuse**: Features are concatenated and projected.
4.  **Shared Encode**: Fused features pass through shared blocks.
    *   **Gradient Stopping**: The forward pass splits at layer $L-K$. A detached copy of the features is created for the contrastive loss, ensuring gradients from this loss only update the top $K$ layers, while the original path continues for reconstruction.

## Datasets

This codebase supports the datasets detailed and referenced in the paper:

1.  **THz Dataset (NE-Data)**:
    ```bibtex
    @article{thz-dataset,
    title = {Data signals for deep learning applications in Terahertz Commun.},
    journal = {Computer Networks},
    volume = {254},
    pages = {110800},
    year = {2024},
    issn = {1389-1286},
    doi = {https://doi.org/10.1016/j.comnet.2024.110800},
    author = {Duschia Bodet and Jacob Hall and Ahmad Masihi and Ngwe Thawdar and Tommaso Melodia and Francesco Restuccia and Josep M. Jornet},
    }
    ```

2.  **ITALYSIG**: 
    ```bibtex
    @online{italysig,
      author       = {L. M. Monteforte and L. Chiaraviglio},
      title        = {ITALYSIG: Open National Database of I/Q Captures},
      year         = {2024},
      url          = {https://italysig.netgroup.uniroma2.it/s/pkjdnMiKMkXQwCe},
      note         = {Accessed: Jul. 11, 2025}
    }
    ```

*Note: This repository contains only the code. Please refer to the references for data access.*
