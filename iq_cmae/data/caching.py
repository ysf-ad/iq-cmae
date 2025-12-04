import os
import hashlib
import json
import numpy as np
from typing import Optional, Dict, Any

class DataCache:
    def __init__(self, cache_dir: Optional[str]):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_filename(self, file_path: str, suffix: str, ext: str) -> Optional[str]:
        """Get cache path for a modality."""
        if not self.cache_dir:
            return None

        # Use stable, deterministic hash for cache filenames across runs
        file_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"{file_hash}_{suffix}.{ext}")

    def get_path(self, file_path: str, modality: str) -> Optional[str]:
        return self._cache_filename(file_path, modality, 'npy')

    def get_metadata_path(self, file_path: str) -> Optional[str]:
        return self._cache_filename(file_path, 'meta', 'json')

    def load(self, cache_path: str) -> Optional[np.ndarray]:
        """Load modality from cache if available."""
        if not cache_path:
            return None
        try:
            # Use uncompressed .npy for SPEED (decompression is slow)
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                return data
            # Fallback: check for old compressed format
            compressed_path = cache_path.replace('.npy', '.npz')
            if os.path.exists(compressed_path):
                loaded = np.load(compressed_path)
                data = loaded['arr_0'] if 'arr_0' in loaded else loaded[list(loaded.keys())[0]]
                loaded.close()
                # Convert back to float32 for consistency
                return data.astype(np.float32)
            return None
        except Exception:
            # File doesn't exist or other error - return None silently
            return None
    
    def save(self, cache_path: str, data: np.ndarray):
        """Save modality to cache - OPTIMIZED FOR SPEED."""
        if cache_path:
            try:
                # Use uncompressed .npy with float32 for MAXIMUM SPEED
                np.save(cache_path, data.astype(np.float32))
                
                # Clean up old compressed cache if it exists
                compressed_path = cache_path.replace('.npy', '.npz')
                if os.path.exists(compressed_path):
                    try:
                        os.remove(compressed_path)
                    except:
                        pass
            except Exception:
                pass

    def load_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        meta_path = self.get_metadata_path(file_path)
        if not meta_path:
            return None
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def save_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        meta_path = self.get_metadata_path(file_path)
        if meta_path:
            try:
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
            except Exception:
                pass
