import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from torch.utils.data import Dataset
from utils.iq_extractor import extract_iq_data
from .caching import DataCache
from .transforms import (
    create_spectrogram_custom,
    create_constellation_image,
    create_gaf_image,
    trim_leading_zeros,
    expand_short_samples
)

class ItalySigRawDataset(Dataset):
    """
    Dataset for loading raw SIGMF files from the ITALYSIG dataset.
    Modernized to support on-the-fly noise generation and multi-modal transforms.
    """
    
    def __init__(
        self,
        data_root: str,
        image_size: int = 224,
        spectrogram_params: Optional[Dict[str, Any]] = None,
        constellation_params: Optional[Dict[str, Any]] = None,
        gaf_params: Optional[Dict[str, Any]] = None,
        trim_leading_zeros: bool = True,
        expand_short_samples: bool = True,
        min_length: int = 224,
        teacher_noise_std: float = 0.0,
        teacher_noise_snr_db: Optional[float] = None,
        student_noise_std: float = 0.0,
        student_noise_snr_db: Optional[float] = None,
        subset_ratio: float = 1.0,
        seed: int = 42,
        modality_mask: Optional[str] = None,
        cache_dir: Optional[str] = None,
        split: str = "train", # train, val, test, or all
        class_map: Optional[Dict[str, int]] = None,
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.trim_leading_zeros_flag = trim_leading_zeros
        self.expand_short_samples_flag = expand_short_samples
        self.min_length = min_length
        
        # Noise configuration
        self.teacher_noise_std = teacher_noise_std
        self.teacher_noise_snr_db = teacher_noise_snr_db
        self.student_noise_std = student_noise_std
        self.student_noise_snr_db = student_noise_snr_db
        
        # Modality configuration
        self._configure_modalities(modality_mask)
        
        # Processing parameters
        self.spectrogram_params = {
            'nperseg': 192, 'noverlap': 144, 'nfft': 384, 'window_type': 'blackman'
        }
        if spectrogram_params: self.spectrogram_params.update(spectrogram_params)
            
        self.constellation_params = {'image_size': image_size, 'range_val': 2.0, 'bins': image_size}
        if constellation_params: self.constellation_params.update(constellation_params)
            
        self.gaf_params = {'image_size': image_size, 'clip_range': 3.0}
        if gaf_params: self.gaf_params.update(gaf_params)
        
        # Initialize Cache
        self.cache = DataCache(cache_dir)

        # Discover Samples
        all_samples = self._discover_samples(data_root)
        
        # Apply Stratified Split
        self.samples = self._apply_stratified_split(all_samples, split, seed)
        
        # Subset Sampling (after split, for debugging/small scale)
        if subset_ratio < 1.0:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.samples), size=int(len(self.samples) * subset_ratio), replace=False)
            self.samples = [self.samples[i] for i in indices]

        # Class Mapping
        self._build_class_map(class_map)

    def _apply_stratified_split(self, samples: List[Dict[str, Any]], split: str, seed: int) -> List[Dict[str, Any]]:
        """Apply stratified 70/15/15 split based on label."""
        if split == "all":
            return samples
            
        # Group by label
        grouped = {}
        for s in samples:
            grouped.setdefault(s['label_str'], []).append(s)
            
        # Sort for determinism
        for k in grouped:
            grouped[k].sort(key=lambda x: x['file_path'])
            
        selected_samples = []
        for label, group in grouped.items():
            n = len(group)
            if n == 0: continue
            
            # Stable shuffle
            group_seed = seed + hash(label) % (2**32)
            rng = np.random.default_rng(group_seed)
            indices = np.arange(n)
            rng.shuffle(indices)
            
            n_train = int(round(n * 0.70))
            n_val = int(round(n * 0.15))
            
            if split == "train":
                subset_indices = indices[:n_train]
            elif split == "val":
                subset_indices = indices[n_train:n_train+n_val]
            elif split == "test":
                subset_indices = indices[n_train+n_val:]
            else:
                raise ValueError(f"Unknown split: {split}")
                
            for i in subset_indices:
                selected_samples.append(group[i])
                
        return selected_samples

    def _discover_samples(self, root: str) -> List[Dict[str, Any]]:
        """Recursively find all .sigmf-data files."""
        samples = []
        root_path = Path(root)
        
        for file_path in root_path.rglob('*.sigmf-data'):
            # Assume parent folder is class name for now, or use metadata
            # ITALYSIG structure might vary, but let's assume class/file.sigmf-data
            # or just use the folder name as the label
            label = file_path.parent.name
            
            samples.append({
                'file_path': str(file_path),
                'meta_path': str(file_path.with_suffix('.sigmf-meta')),
                'label_str': label,
                'noise_seed': hash(str(file_path)) % (2**32) # Deterministic seed per file
            })
            
        return sorted(samples, key=lambda x: x['file_path'])

    def _configure_modalities(self, modality_mask: Optional[str]):
        if modality_mask is None or modality_mask.lower() == "all":
            self.include_constellation = True
            self.include_gaf = True
            self.include_spectrogram = True
        else:
            tokens = [t.strip().lower() for t in str(modality_mask).replace(",", "+").split("+") if t.strip()]
            self.include_constellation = "constellation" in tokens or "constellation_only" in tokens
            self.include_gaf = "gaf" in tokens or "gaf_only" in tokens
            self.include_spectrogram = "spectrogram" in tokens or "spectrogram_only" in tokens

    def _build_class_map(self, class_map: Optional[Dict[str, int]]):
        if class_map is not None:
            self.class_to_idx = class_map
        else:
            class_names = sorted({s['label_str'] for s in self.samples})
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        file_path = sample['file_path']
        
        # Load I/Q Data
        iq_data, meta_info = self._load_sigmf_file(file_path, sample['meta_path'])
        
        # Pre-processing
        if self.trim_leading_zeros_flag:
            iq_data = trim_leading_zeros(iq_data)
        if self.expand_short_samples_flag:
            iq_data = expand_short_samples(iq_data, self.min_length)

        # --- Generate Teacher (Clean or Teacher Noise) ---
        teacher_iq = iq_data
        if self.teacher_noise_std > 0 or self.teacher_noise_snr_db is not None:
            teacher_iq = self._add_iq_noise(iq_data, sample, is_teacher=True)
            teacher_image = self._generate_modalities(teacher_iq, file_path, use_cache=False)
        else:
            teacher_image = self._generate_modalities(teacher_iq, file_path, use_cache=True)

        # --- Generate Student (Clean or Student Noise) ---
        student_iq = iq_data
        if self.student_noise_std > 0 or self.student_noise_snr_db is not None:
            student_iq = self._add_iq_noise(iq_data, sample, is_teacher=False)
            student_image = self._generate_modalities(student_iq, file_path, use_cache=False)
        else:
            if self.teacher_noise_std == 0 and self.teacher_noise_snr_db is None:
                 student_image = teacher_image
            else:
                 student_image = self._generate_modalities(student_iq, file_path, use_cache=True)

        # Prepare Output
        label_idx = self.class_to_idx.get(sample['label_str'], 0)

        return {
            'image': torch.from_numpy(student_image).float(),
            'teacher_image': torch.from_numpy(teacher_image).float(),
            'label': label_idx,
            'sample_idx': idx,
            'file_path': file_path,
            'meta_info': meta_info
        }

    def _generate_modalities(self, iq_data: np.ndarray, file_path: str, use_cache: bool) -> np.ndarray:
        """Generate and combine modalities from I/Q data."""
        modalities = {}
        
        # 1. Spectrogram
        if self.include_spectrogram:
            spec = None
            if use_cache: spec = self.cache.load(self.cache.get_path(file_path, 'spectrogram'))
            if spec is None:
                spec = create_spectrogram_custom(iq_data, self.spectrogram_params, self.image_size)
                if use_cache: self.cache.save(self.cache.get_path(file_path, 'spectrogram'), spec)
            modalities['spectrogram'] = spec

        # 2. Constellation
        if self.include_constellation:
            const = None
            if use_cache: const = self.cache.load(self.cache.get_path(file_path, 'constellation'))
            if const is None:
                const = create_constellation_image(iq_data, self.constellation_params['image_size'])
                if use_cache: self.cache.save(self.cache.get_path(file_path, 'constellation'), const)
            modalities['constellation'] = const

        # 3. GAF
        if self.include_gaf:
            gaf = None
            if use_cache: gaf = self.cache.load(self.cache.get_path(file_path, 'gaf'))
            if gaf is None:
                gaf = create_gaf_image(iq_data, self.gaf_params['image_size'], self.gaf_params['clip_range'])
                if use_cache: self.cache.save(self.cache.get_path(file_path, 'gaf'), gaf)
            modalities['gaf'] = gaf

        return self._combine_modalities(modalities)

    def _combine_modalities(self, modalities: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        if self.include_constellation: parts.append(modalities['constellation'])
        if self.include_gaf: parts.append(modalities['gaf'])
        if self.include_spectrogram: parts.append(modalities['spectrogram'])
        
        if not parts: return np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def _add_iq_noise(self, iq_data: np.ndarray, sample: Dict[str, Any], is_teacher: bool) -> np.ndarray:
        """Add Gaussian noise to I/Q data."""
        std = self.teacher_noise_std if is_teacher else self.student_noise_std
        snr = self.teacher_noise_snr_db if is_teacher else self.student_noise_snr_db

        if std <= 0 and snr is None: return iq_data

        seed = sample['noise_seed']
        if not is_teacher: seed += 1
        
        rng_gen = np.random.default_rng(seed)
        noise = rng_gen.standard_normal(iq_data.shape).astype(np.float32)
        
        scale = 1.0
        if snr is not None:
            signal_power = np.mean(iq_data**2)
            noise_power = signal_power / (10**(snr/10))
            scale = np.sqrt(noise_power)
        if subset_ratio < 1.0:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.samples), size=int(len(self.samples) * subset_ratio), replace=False)
            self.samples = [self.samples[i] for i in indices]

        # Class Mapping
        self._build_class_map(class_map)

    def _apply_stratified_split(self, samples: List[Dict[str, Any]], split: str, seed: int) -> List[Dict[str, Any]]:
        """Apply stratified 70/15/15 split based on label."""
        if split == "all":
            return samples
            
        # Group by label
        grouped = {}
        for s in samples:
            grouped.setdefault(s['label_str'], []).append(s)
            
        # Sort for determinism
        for k in grouped:
            grouped[k].sort(key=lambda x: x['file_path'])
            
        selected_samples = []
        for label, group in grouped.items():
            n = len(group)
            if n == 0: continue
            
            # Stable shuffle
            group_seed = seed + hash(label) % (2**32)
            rng = np.random.default_rng(group_seed)
            indices = np.arange(n)
            rng.shuffle(indices)
            
            n_train = int(round(n * 0.70))
            n_val = int(round(n * 0.15))
            
            if split == "train":
                subset_indices = indices[:n_train]
            elif split == "val":
                subset_indices = indices[n_train:n_train+n_val]
            elif split == "test":
                subset_indices = indices[n_train+n_val:]
            else:
                raise ValueError(f"Unknown split: {split}")
                
            for i in subset_indices:
                selected_samples.append(group[i])
                
        return selected_samples

    def _discover_samples(self, root: str) -> List[Dict[str, Any]]:
        """Recursively find all .sigmf-data files."""
        samples = []
        root_path = Path(root)
        
        for file_path in root_path.rglob('*.sigmf-data'):
            # Assume parent folder is class name for now, or use metadata
            # ITALYSIG structure might vary, but let's assume class/file.sigmf-data
            # or just use the folder name as the label
            label = file_path.parent.name
            
            samples.append({
                'file_path': str(file_path),
                'meta_path': str(file_path.with_suffix('.sigmf-meta')),
                'label_str': label,
                'noise_seed': hash(str(file_path)) % (2**32) # Deterministic seed per file
            })
            
        return sorted(samples, key=lambda x: x['file_path'])

    def _configure_modalities(self, modality_mask: Optional[str]):
        if modality_mask is None or modality_mask.lower() == "all":
            self.include_constellation = True
            self.include_gaf = True
            self.include_spectrogram = True
        else:
            tokens = [t.strip().lower() for t in str(modality_mask).replace(",", "+").split("+") if t.strip()]
            self.include_constellation = "constellation" in tokens or "constellation_only" in tokens
            self.include_gaf = "gaf" in tokens or "gaf_only" in tokens
            self.include_spectrogram = "spectrogram" in tokens or "spectrogram_only" in tokens

    def _build_class_map(self, class_map: Optional[Dict[str, int]]):
        if class_map is not None:
            self.class_to_idx = class_map
        else:
            class_names = sorted({s['label_str'] for s in self.samples})
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        file_path = sample['file_path']
        
        # Load I/Q Data
        iq_data, meta_info = self._load_sigmf_file(file_path, sample['meta_path'])
        
        # Pre-processing
        if self.trim_leading_zeros_flag:
            iq_data = trim_leading_zeros(iq_data)
        if self.expand_short_samples_flag:
            iq_data = expand_short_samples(iq_data, self.min_length)

        # --- Generate Teacher (Clean or Teacher Noise) ---
        teacher_iq = iq_data
        if self.teacher_noise_std > 0 or self.teacher_noise_snr_db is not None:
            teacher_iq = self._add_iq_noise(iq_data, sample, is_teacher=True)
            teacher_image = self._generate_modalities(teacher_iq, file_path, use_cache=False)
        else:
            teacher_image = self._generate_modalities(teacher_iq, file_path, use_cache=True)

        # --- Generate Student (Clean or Student Noise) ---
        student_iq = iq_data
        if self.student_noise_std > 0 or self.student_noise_snr_db is not None:
            student_iq = self._add_iq_noise(iq_data, sample, is_teacher=False)
            student_image = self._generate_modalities(student_iq, file_path, use_cache=False)
        else:
            if self.teacher_noise_std == 0 and self.teacher_noise_snr_db is None:
                 student_image = teacher_image
            else:
                 student_image = self._generate_modalities(student_iq, file_path, use_cache=True)

        # Prepare Output
        label_idx = self.class_to_idx.get(sample['label_str'], 0)

        return {
            'image': torch.from_numpy(student_image).float(),
            'teacher_image': torch.from_numpy(teacher_image).float(),
            'label': label_idx,
            'sample_idx': idx,
            'file_path': file_path,
            'meta_info': meta_info
        }

    def _generate_modalities(self, iq_data: np.ndarray, file_path: str, use_cache: bool) -> np.ndarray:
        """Generate and combine modalities from I/Q data."""
        modalities = {}
        
        # 1. Spectrogram
        if self.include_spectrogram:
            spec = None
            if use_cache: spec = self.cache.load(self.cache.get_path(file_path, 'spectrogram'))
            if spec is None:
                spec = create_spectrogram_custom(iq_data, self.spectrogram_params, self.image_size)
                if use_cache: self.cache.save(self.cache.get_path(file_path, 'spectrogram'), spec)
            modalities['spectrogram'] = spec

        # 2. Constellation
        if self.include_constellation:
            const = None
            if use_cache: const = self.cache.load(self.cache.get_path(file_path, 'constellation'))
            if const is None:
                const = create_constellation_image(iq_data, self.constellation_params['image_size'])
                if use_cache: self.cache.save(self.cache.get_path(file_path, 'constellation'), const)
            modalities['constellation'] = const

        # 3. GAF
        if self.include_gaf:
            gaf = None
            if use_cache: gaf = self.cache.load(self.cache.get_path(file_path, 'gaf'))
            if gaf is None:
                gaf = create_gaf_image(iq_data, self.gaf_params['image_size'], self.gaf_params['clip_range'])
                if use_cache: self.cache.save(self.cache.get_path(file_path, 'gaf'), gaf)
            modalities['gaf'] = gaf

        return self._combine_modalities(modalities)

    def _combine_modalities(self, modalities: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        if self.include_constellation: parts.append(modalities['constellation'])
        if self.include_gaf: parts.append(modalities['gaf'])
        if self.include_spectrogram: parts.append(modalities['spectrogram'])
        
        if not parts: return np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def _add_iq_noise(self, iq_data: np.ndarray, sample: Dict[str, Any], is_teacher: bool) -> np.ndarray:
        """Add Gaussian noise to I/Q data."""
        std = self.teacher_noise_std if is_teacher else self.student_noise_std
        snr = self.teacher_noise_snr_db if is_teacher else self.student_noise_snr_db

        if std <= 0 and snr is None: return iq_data

        seed = sample['noise_seed']
        if not is_teacher: seed += 1
        
        rng_gen = np.random.default_rng(seed)
        noise = rng_gen.standard_normal(iq_data.shape).astype(np.float32)
        
        scale = 1.0
        if snr is not None:
            signal_power = np.mean(iq_data**2)
            noise_power = signal_power / (10**(snr/10))
            scale = np.sqrt(noise_power)
        else:
            scale = std 

        return iq_data + noise * scale

    def _load_sigmf_file(self, file_path: str, meta_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            full_iq = extract_iq_data(file_path, meta_path)
            iq_matrix = full_iq.T.astype(np.float32)
        except:
            # Fallback for raw complex64
            with open(file_path, 'rb') as f:
                iq_data = np.frombuffer(f.read(), dtype=np.complex64)
            iq_matrix = np.stack([np.real(iq_data), np.imag(iq_data)], axis=0).astype(np.float32)

        return iq_matrix, {'sample_count': iq_matrix.shape[1]}
