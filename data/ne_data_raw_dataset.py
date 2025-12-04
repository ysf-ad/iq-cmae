#!/usr/bin/env python3
"""
NE-Data Raw Dataset

On-the-fly dataset for loading and processing SIGMF files from the ne-data directory.
Refactored to use modular components for discovery, caching, and transforms.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from torch.utils.data import Dataset
from utils.iq_extractor import extract_iq_data, parse_sigmf_meta
from .discovery import discover_samples, apply_voltage_split, apply_subset_sampling, compute_noise_seed
from .caching import DataCache
from .transforms import (
    create_spectrogram_custom,
    create_constellation_image,
    create_gaf_image,
    trim_leading_zeros,
    expand_short_samples
)

class NEDataRawDataset(Dataset):
    """Dataset for loading raw SIGMF files from ne-data directory."""
    
    def __init__(
        self,
        data_root: str,
        bandwidth: Optional[str] = "5 GHz Bandwidth",
        power_levels: Optional[List[str]] = None,
        modulation_types: Optional[List[str]] = None,
        symbol_rates: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        file_list: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        image_size: int = 224,
        spectrogram_params: Optional[Dict[str, Any]] = None,
        constellation_params: Optional[Dict[str, Any]] = None,
        gaf_params: Optional[Dict[str, Any]] = None,
        trim_leading_zeros: bool = True,
        expand_short_samples: bool = True,
        min_length: int = 224,
        spectrogram_style: str = "zold",
        teacher_noise_std: float = 0.0,
        teacher_noise_snr_db: Optional[float] = None,
        teacher_noise_std_range: Optional[Tuple[float, float]] = None,
        student_noise_std: float = 0.0,
        student_noise_snr_db: Optional[float] = None,
        student_noise_std_range: Optional[Tuple[float, float]] = None,
        noise_override: Optional[Dict[str, Any]] = None,
        subset_ratio: float = 1.0,
        seed: int = 42,
        exclude_train_ratio: Optional[float] = None,
        voltage_split: str = "all",
        noise_seed_mode: str = "deterministic",
        scale_noise_by_power: bool = True,
        power_reference_level: Optional[str] = None,
        modality_mask: Optional[str] = None,
        in_memory: bool = False,
        label_mode: str = "modulation",
        class_map: Optional[Dict[str, int]] = None,
    ):
        self.data_root = data_root
        self.bandwidth = bandwidth
        self.power_levels = power_levels or ["600mV", "75mV"]
        self.modulation_types = modulation_types or ["16QAM", "4PSK", "64QAM", "8PSK"]
        self.symbol_rates = symbol_rates
        self.image_size = image_size
        self.trim_leading_zeros_flag = trim_leading_zeros
        self.expand_short_samples_flag = expand_short_samples
        self.min_length = min_length
        self.spectrogram_style = "zold"
        self.label_mode = label_mode
        self.in_memory = in_memory
        self.memory_cache = {}

        # Noise configuration
        self.teacher_noise_std = teacher_noise_std
        self.teacher_noise_snr_db = teacher_noise_snr_db
        self.teacher_noise_std_range = teacher_noise_std_range
        self.student_noise_std = student_noise_std
        self.student_noise_snr_db = student_noise_snr_db
        self.student_noise_std_range = student_noise_std_range
        
        if noise_override:
            self.teacher_noise_std = noise_override.get('teacher_noise_std', self.teacher_noise_std)
            self.teacher_noise_snr_db = noise_override.get('teacher_noise_snr_db', self.teacher_noise_snr_db)
            self.student_noise_std = noise_override.get('student_noise_std', self.student_noise_std)
            self.student_noise_snr_db = noise_override.get('student_noise_snr_db', self.student_noise_snr_db)

        self._disable_teacher_cache = (self.teacher_noise_std_range is not None)
        self.noise_seed_mode = noise_seed_mode
        self.scale_noise_by_power = scale_noise_by_power
        
        # Power level scaling
        self.power_level_values = self._compute_power_level_values(self.power_levels)
        self.power_reference_value = self._parse_power_level(power_reference_level) if power_reference_level else (max(self.power_level_values.values()) if self.power_level_values else 1.0)
        
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

        # Discover and Split Samples
        self.samples = discover_samples(
            data_root, bandwidth, self.power_levels, self.modulation_types, 
            self.symbol_rates, file_list, max_samples_per_class
        )
        # self.samples = discover_samples(...)

        if voltage_split != "all":
            self.samples = apply_voltage_split(self.samples, voltage_split, seed)
        
        if subset_ratio < 1.0:
            self.samples = apply_subset_sampling(self.samples, subset_ratio, seed, exclude_train_ratio)

        # Class Mapping
        self._build_class_map(class_map)

        # Load to memory if requested
        if self.in_memory:
            self._load_all_to_memory()

    def _configure_modalities(self, modality_mask: Optional[str]):
        if modality_mask is None or modality_mask.lower() == "all":
            self.include_constellation = True
            self.include_gaf = True
            self.include_spectrogram = True
        else:
            tokens = [t.strip().lower() for t in str(modality_mask).replace(",", "+").split("+") if t.strip()]
            token_map = {
                "constellation_only": "constellation",
                "gaf_only": "gaf",
                "spectrogram_only": "spectrogram",
            }
            tokens = [token_map.get(t, t) for t in tokens]
            self.include_constellation = "constellation" in tokens
            self.include_gaf = "gaf" in tokens
            self.include_spectrogram = "spectrogram" in tokens

    def _build_class_map(self, class_map: Optional[Dict[str, int]]):
        if class_map is not None:
            self.class_to_idx = class_map
        else:
            if self.label_mode == "fine_grained":
                class_names = sorted({self._get_sample_label_name(s) for s in self.samples})
            else:
                class_names = sorted({s['modulation'] for s in self.samples})
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def _load_all_to_memory(self):
        from tqdm import tqdm
        for i in tqdm(range(len(self)), desc="Loading to RAM"):
            _ = self[i]

    def _parse_power_level(self, power_level: Optional[str]) -> float:
        if not power_level: return 1.0
        try:
            return max(float(power_level.strip().lower().replace("mv", "")), 1e-6)
        except: return 1.0

    def _compute_power_level_values(self, power_levels: List[str]) -> Dict[str, float]:
        return {level: self._parse_power_level(level) for level in power_levels}
    
    def _power_level_scale(self, power_level: Optional[str]) -> float:
        if not self.scale_noise_by_power or self.teacher_noise_snr_db is not None:
            return 1.0
        val = self.power_level_values.get(power_level, self.power_reference_value)
        return float(val) / float(self.power_reference_value)

    def _get_sample_label_name(self, sample: Dict[str, Any]) -> str:
        if self.label_mode == "fine_grained":
            bw = self.bandwidth.replace(" ", "")
            return f"{bw}_{sample['modulation']}_{sample['power_level']}"
        return sample['modulation']

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self.memory_cache:
            return self.memory_cache[idx]

        sample = self.samples[idx]
        file_path = sample['file_path']
        
        # Load I/Q Data
        iq_data, meta_info = self._load_sigmf_file(file_path)
        
        # Pre-processing
        if self.trim_leading_zeros_flag:
            iq_data = trim_leading_zeros(iq_data)
        if self.expand_short_samples_flag:
            iq_data = expand_short_samples(iq_data, self.min_length)

        # --- Generate Teacher (Clean or Teacher Noise) ---
        teacher_iq = iq_data
        if self.teacher_noise_std > 0 or self.teacher_noise_snr_db is not None or self.teacher_noise_std_range is not None:
            teacher_iq = self._add_iq_noise(iq_data, sample, is_teacher=True)
            teacher_image = self._generate_modalities(teacher_iq, file_path, use_cache=False)
        else:
            teacher_image = self._generate_modalities(teacher_iq, file_path, use_cache=True)

        # --- Generate Student (Clean or Student Noise) ---
        student_iq = iq_data
        if self.student_noise_std > 0 or self.student_noise_snr_db is not None or self.student_noise_std_range is not None:
            student_iq = self._add_iq_noise(iq_data, sample, is_teacher=False)
            student_image = self._generate_modalities(student_iq, file_path, use_cache=False)
        else:
            if self.teacher_noise_std == 0 and self.teacher_noise_snr_db is None and self.teacher_noise_std_range is None:
                 student_image = teacher_image
            else:
                 student_image = self._generate_modalities(student_iq, file_path, use_cache=True)

        # Prepare Output
        label_name = self._get_sample_label_name(sample)
        label_idx = self.class_to_idx.get(label_name, 0)

        result = {
            'image': torch.from_numpy(student_image).float(),
            'teacher_image': torch.from_numpy(teacher_image).float(),
            'label': label_idx,
            'sample_idx': idx,
            'file_path': file_path,
            'meta_info': meta_info
        }

        if self.in_memory:
            self.memory_cache[idx] = result
            
        return result

    def _generate_modalities(self, iq_data: np.ndarray, file_path: str, use_cache: bool) -> np.ndarray:
        """Generate and combine modalities from I/Q data."""
        modalities = {}
        
        # 1. Spectrogram
        if self.include_spectrogram:
            spec = None
            if use_cache:
                spec = self.cache.load(self.cache.get_path(file_path, 'spectrogram'))
            
            if spec is None:
                spec = create_spectrogram_custom(iq_data, self.spectrogram_params, self.image_size)
                if use_cache:
                    self.cache.save(self.cache.get_path(file_path, 'spectrogram'), spec)
            modalities['spectrogram'] = spec

        # 2. Constellation
        if self.include_constellation:
            const = None
            if use_cache:
                const = self.cache.load(self.cache.get_path(file_path, 'constellation'))
            
            if const is None:
                const = create_constellation_image(iq_data, self.constellation_params['image_size'])
                if use_cache:
                    self.cache.save(self.cache.get_path(file_path, 'constellation'), const)
            modalities['constellation'] = const

        # 3. GAF
        if self.include_gaf:
            gaf = None
            if use_cache:
                gaf = self.cache.load(self.cache.get_path(file_path, 'gaf'))
            
            if gaf is None:
                gaf = create_gaf_image(iq_data, self.gaf_params['image_size'], self.gaf_params['clip_range'])
                if use_cache:
                    self.cache.save(self.cache.get_path(file_path, 'gaf'), gaf)
            modalities['gaf'] = gaf

        return self._combine_modalities(modalities)

    def _combine_modalities(self, modalities: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        if self.include_constellation: parts.append(modalities['constellation'])
        if self.include_gaf: parts.append(modalities['gaf'])
        if self.include_spectrogram: parts.append(modalities['spectrogram'])
        
        if not parts:
            return np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
            
        return np.concatenate(parts, axis=0)

    def _add_iq_noise(self, iq_data: np.ndarray, sample: Dict[str, Any], is_teacher: bool) -> np.ndarray:
        """Add Gaussian noise to I/Q data."""
        # Determine noise parameters
        if is_teacher:
            std = self.teacher_noise_std
            snr = self.teacher_noise_snr_db
            rng = self.teacher_noise_std_range
        else:
            std = self.student_noise_std
            snr = self.student_noise_snr_db
            rng = self.student_noise_std_range

        if rng:
            std = np.random.uniform(rng[0], rng[1])

        if std <= 0 and snr is None:
            return iq_data

        # Generate Noise
        seed = sample['noise_seed'] if self.noise_seed_mode == 'deterministic' else None
        if seed and not is_teacher: seed += 1 # Different seed for student
        
        rng_gen = np.random.default_rng(seed)
        noise = rng_gen.standard_normal(iq_data.shape).astype(np.float32)
        
        # Scale Noise
        scale = 1.0
        if snr is not None:
            # Calculate signal power
            signal_power = np.mean(iq_data**2)
            noise_power = signal_power / (10**(snr/10))
            scale = np.sqrt(noise_power)
        else:
            scale = std * self._power_level_scale(sample['power_level'])

        return iq_data + noise * scale

    def _load_sigmf_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Simplified loader that handles errors gracefully
        meta_path = file_path.replace('.sigmf-data', '.sigmf-meta')
        try:
            full_iq = extract_iq_data(file_path, meta_path)
            iq_matrix = full_iq.T.astype(np.float32)
        except:
            # Fallback
            with open(file_path, 'rb') as f:
                iq_data = np.frombuffer(f.read(), dtype=np.complex64)
            iq_matrix = np.stack([np.real(iq_data), np.imag(iq_data)], axis=0).astype(np.float32)

        return iq_matrix, {'sample_count': iq_matrix.shape[1]}
