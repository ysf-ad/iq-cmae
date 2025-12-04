import os
import glob
import json
import hashlib
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

def _stable_int_hash(*parts: str) -> int:
    """Deterministic hash helper (first 8 hex chars of MD5)."""
    payload = "||".join(parts).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)

def compute_noise_seed(file_path: str) -> int:
    """Compute a deterministic noise seed for a given file path."""
    digest = hashlib.md5(file_path.encode('utf-8')).hexdigest()
    # Use first 8 hex digits to fit within 32-bit range
    return int(digest[:8], 16)

def discover_samples(data_root: str, bandwidth: Optional[str], power_levels: List[str], 
                    modulation_types: List[str], symbol_rates: Optional[List[str]],
                    file_list: Optional[List[str]] = None, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Discover all SIGMF files matching the criteria."""
    samples = []
    
    # When using file_list, we don't need bandwidth_path
    if file_list:
        for rel_path in file_list:
            normalized_rel = rel_path.replace("\\", "/")
            abs_path = os.path.join(data_root, normalized_rel)
            if not os.path.exists(abs_path):
                continue
            if "noise" in os.path.basename(abs_path).lower():
                continue
            parts = normalized_rel.split('/')
            if len(parts) < 2:
                continue
            file_bandwidth = parts[0]
            if bandwidth and file_bandwidth != bandwidth:
                continue
            meta_parts = parts[1].split('_')
            if len(meta_parts) != 3:
                continue
            modulation, symbol_rate, power_level = meta_parts
            if modulation not in modulation_types or power_level not in power_levels:
                continue
            if symbol_rates and symbol_rate not in symbol_rates:
                continue
            samples.append({
                'file_path': abs_path,
                'modulation': modulation,
                'power_level': power_level,
                'symbol_rate': symbol_rate,
                'bandwidth': file_bandwidth,
                'noise_seed': compute_noise_seed(abs_path),
            })
    else:
        if not bandwidth:
            raise ValueError("bandwidth must be specified when not using file_list")
        bandwidth_path = os.path.join(data_root, bandwidth)
        if not os.path.exists(bandwidth_path):
            print(f"Warning: Bandwidth path not found: {bandwidth_path}")
            return samples
        
        for power_level in power_levels:
            for modulation in modulation_types:
                patterns: List[str]
                if symbol_rates:
                    patterns = [f"{modulation}_{symbol}_{power_level}" for symbol in symbol_rates]
                else:
                    patterns = [f"{modulation}_*_{power_level}"]

                for pattern in patterns:
                    search_path = os.path.join(bandwidth_path, pattern)
                    matching_dirs = glob.glob(search_path)

                    dirs_to_process: List[str] = []
                    symbol_rate_map: Dict[str, Optional[str]] = {}

                    for candidate_dir in matching_dirs:
                        symbol_rate = None
                        if symbol_rates and len(symbol_rates) == 1:
                            symbol_rate = symbol_rates[0]
                        else:
                            dir_name = os.path.basename(candidate_dir)
                            dir_parts = dir_name.split('_')
                            for part in dir_parts:
                                if 'GSyms' in part or 'MSyms' in part:
                                    symbol_rate = part
                                    break
                            
                            if symbol_rate is None:
                                # Fallback to meta file
                                for candidate in glob.glob(os.path.join(candidate_dir, "*.sigmf-meta")):
                                    if "noise" in os.path.basename(candidate).lower():
                                        continue
                                    try:
                                        with open(candidate, "r") as f:
                                            metadata = json.load(f)
                                        captures = metadata.get("captures", [])
                                        if captures:
                                            extras = captures[0].get("extras", {})
                                            rate = extras.get("ntia-ex:modulation_symbol_rate", None)
                                            if rate:
                                                symbol_rate = f"{rate}GSyms"
                                                break
                                    except:
                                        pass
                        
                        if symbol_rates and symbol_rate not in symbol_rates:
                            continue
                        dirs_to_process.append(candidate_dir)
                        symbol_rate_map[candidate_dir] = symbol_rate

                    for dir_path in dirs_to_process:
                        symbol_rate = symbol_rate_map.get(dir_path)
                        sigmf_files = [
                            f for f in glob.glob(os.path.join(dir_path, "*.sigmf-data"))
                            if "noise" not in os.path.basename(f).lower()
                        ]
                        if max_samples and len(sigmf_files) > max_samples:
                            sigmf_files = sigmf_files[:max_samples]
                        for file_path in sigmf_files:
                            samples.append({
                                'file_path': file_path,
                                'modulation': modulation,
                                'power_level': power_level,
                                'symbol_rate': symbol_rate,
                                'bandwidth': bandwidth,
                                'noise_seed': compute_noise_seed(file_path),
                            })
    
    return samples

def apply_voltage_split(samples: List[Dict[str, Any]], split_mode: str, seed: int) -> List[Dict[str, Any]]:
    """Apply stratified 70/15/15 split."""
    if split_mode == "all":
        return samples
    
    # Group samples
    grouped: Dict[Tuple[str, str, Optional[str]], List[Dict[str, Any]]] = {}
    for sample in samples:
        symbol_rate = sample.get('symbol_rate', None)
        key = (sample['modulation'], sample['power_level'], symbol_rate)
        grouped.setdefault(key, []).append(sample)
    
    # Sort for deterministic splitting
    for key in grouped:
        grouped[key].sort(key=lambda s: s['file_path'])
    
    train_samples = []
    eval_train_samples = []
    eval_test_samples = []
    
    for (modulation, power_level, symbol_rate), group_samples in grouped.items():
        n = len(group_samples)
        if n == 0:
            continue
        
        n_train = int(round(n * 0.70))
        n_eval_train = int(round(n * 0.15))
        
        symbol_rate_str = str(symbol_rate) if symbol_rate is not None else "None"
        group_seed = seed + _stable_int_hash(modulation, power_level, symbol_rate_str)
        group_rng = random.Random(group_seed)
        shuffled = group_samples.copy()
        group_rng.shuffle(shuffled)
        
        train_samples.extend(shuffled[:n_train])
        eval_train_samples.extend(shuffled[n_train:n_train + n_eval_train])
        eval_test_samples.extend(shuffled[n_train + n_eval_train:])
    
    if split_mode == "train":
        return train_samples
    elif split_mode in ("eval", "eval_train"):
        return eval_train_samples
    elif split_mode == "eval_test":
        return eval_test_samples
    else:
        return samples

def apply_subset_sampling(samples: List[Dict[str, Any]], subset_ratio: float, seed: int, 
                         exclude_train_ratio: Optional[float] = None) -> List[Dict[str, Any]]:
    """Apply stratified subset sampling."""
    if subset_ratio >= 1.0 and exclude_train_ratio is None:
        return samples
    
    samples_by_group = {}
    for sample in samples:
        symbol_rate = sample.get('symbol_rate', None)
        key = (sample['modulation'], sample['power_level'], symbol_rate)
        samples_by_group.setdefault(key, []).append(sample)
    
    subset_samples = []
    total_samples = len(samples)
    target_eval_samples = int(total_samples * subset_ratio)
    
    for (modulation, power_level, symbol_rate), group_samples in samples_by_group.items():
        group_total = len(group_samples)
        if group_total == 0:
            continue
        
        group_target_eval = max(1, int(target_eval_samples * (group_total / total_samples)))
        
        sorted_samples = sorted(group_samples, key=lambda s: s['file_path'])
        symbol_rate_str = str(symbol_rate) if symbol_rate is not None else "None"
        group_seed = seed + _stable_int_hash(modulation, power_level, symbol_rate_str)
        group_rng = np.random.default_rng(group_seed)
        
        if exclude_train_ratio is not None and exclude_train_ratio > 0:
            n_train = max(1, int(group_total * exclude_train_ratio))
            if n_train >= group_total:
                continue
            
            train_indices = set(group_rng.choice(group_total, size=n_train, replace=False))
            remaining_indices = [i for i in range(group_total) if i not in train_indices]
            remaining_samples = [sorted_samples[i] for i in remaining_indices]
            
            n_eval = min(group_target_eval, len(remaining_samples))
            if n_eval > 0:
                eval_group_seed = group_seed + 10000
                eval_rng = np.random.default_rng(eval_group_seed)
                if n_eval < len(remaining_samples):
                    eval_indices = eval_rng.choice(len(remaining_samples), size=n_eval, replace=False)
                    subset_samples.extend([remaining_samples[i] for i in eval_indices])
                else:
                    subset_samples.extend(remaining_samples)
        else:
            n_samples = max(1, int(group_total * subset_ratio))
            if n_samples < group_total:
                selected_indices = group_rng.choice(group_total, size=n_samples, replace=False)
                subset_samples.extend([sorted_samples[i] for i in selected_indices])
            else:
                subset_samples.extend(sorted_samples)
    
    return subset_samples
