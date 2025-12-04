import torch
import numpy as np
import os
from scipy import signal
from typing import Optional, Dict, Any
from pyts.image import GramianAngularField

# Cache for kernels
_exp_kernel_cache = {}

def create_exponential_kernel(size: int, decay_rate: float = 1.0) -> np.ndarray:
    """Create a 2D exponential decay kernel using scipy.signal.windows.exponential."""
    cache_key = (size, decay_rate)
    if cache_key in _exp_kernel_cache:
        return _exp_kernel_cache[cache_key]
    from scipy.signal.windows import exponential

    tau = 1.0 / decay_rate 
    # Create 1D exponential window
    window_1d = exponential(size, tau=tau, sym=True)
    # Create 2D kernel by outer product of 1D windows
    kernel = np.outer(window_1d, window_1d)
    # Normalize to sum to 1
    kernel = kernel / kernel.sum()
    _exp_kernel_cache[cache_key] = kernel
    return kernel

def create_constellation_image(iq_data: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Create constellation density image from I/Q data.

    Pipeline:
    1) Build 2D histogram over fixed range (±2.0 std)
    2) Apply exponential smoothing (rate=1.0) as a base denoiser
    3) Create 3 channels via Gaussian blurs with σ = 1.0, 2.0, 4.0
    4) Per-channel normalize to [0,1]; return float32 (3, H, W)
    """
    if iq_data.shape[0] != 2:
        if iq_data.shape[1] == 2:
            iq_data = iq_data.T
        else:
            raise ValueError(f"iq_data must be 2xN or Nx2, got {iq_data.shape}")

    I = iq_data[0]
    Q = iq_data[1]

    # Fixed range based on standard deviation
    range_val = 2.0
    bins = image_size

    # 1) 2D Histogram
    H, _, _ = np.histogram2d(I, Q, bins=bins, range=[[-range_val, range_val], [-range_val, range_val]])
    H = H.T  # Transpose to match image coordinates

    # 2) Exponential Smoothing
    kernel_size = 5
    kernel = create_exponential_kernel(kernel_size, decay_rate=1.0)
    H_smooth = signal.convolve2d(H, kernel, mode='same')

    # 3) Multi-scale Gaussian Blur (3 channels)
    from scipy.ndimage import gaussian_filter
    sigmas = [1.0, 2.0, 4.0]
    channels = []
    for sigma in sigmas:
        ch = gaussian_filter(H_smooth, sigma=sigma)
        # Normalize per channel
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch = np.zeros_like(ch)
        channels.append(ch)

    img = np.stack(channels, axis=0)  # (3, H, W)
    return img.astype(np.float32)

def create_gaf_image(iq_data: np.ndarray, image_size: int = 224, clip_range: float = 3.0) -> np.ndarray:
    """Create a 2-channel GASF image from I and Q with robust preprocessing.

    Pipeline per channel (I and Q):
    - DC removal and std normalization
    - Resample to image_size (decimate if longer, interpolate if shorter)
    - Clip to [-3, 3] and scale to [-1, 1]
    - Apply pyts GramianAngularField(method='summation')

    Returns (2, H, W) float32.
    """
    if iq_data.shape[0] != 2:
        if iq_data.shape[1] == 2:
            iq_data = iq_data.T

    I = iq_data[0].astype(np.float32)
    Q = iq_data[1].astype(np.float32)

    def preprocess(vec: np.ndarray) -> np.ndarray:
        v = vec - float(np.mean(vec))
        std = float(np.std(v))
        if std > 0:
            v = v / std
        n = v.shape[0]
        if n != image_size:
            if n > image_size:
                step = max(1, n // image_size)
                v = v[::step][:image_size]
            else:
                idx = np.linspace(0, n - 1, image_size, dtype=int)
                v = v[idx]
        v = np.clip(v, -clip_range, clip_range) / clip_range  # [-1,1]
        return v.astype(np.float32)

    I_p = preprocess(I)
    Q_p = preprocess(Q)

    gaf = GramianAngularField(image_size=image_size, method='summation')
    G = gaf.fit_transform(np.vstack([I_p, Q_p]))  # (2, H, W)
    
    # Normalize per channel
    out = np.empty_like(G, dtype=np.float32)
    for i in range(G.shape[0]):
        g = G[i]
        g_min, g_max = g.min(), g.max()
        if g_max > g_min:
            out[i] = (g - g_min) / (g_max - g_min + 1e-8)
        else:
            out[i] = np.zeros_like(g)
            
    return out.astype(np.float32)

def create_spectrogram_custom(iq_data: np.ndarray, params: dict, image_size: int) -> np.ndarray:
    """Create spectrogram with custom parameters."""
    if iq_data.shape[0] != 2:
        if iq_data.shape[1] == 2:
            iq_data = iq_data.T
    
    i_component = iq_data[0]
    nperseg = params.get('nperseg', 192)
    noverlap = params.get('noverlap', 144)
    nfft = params.get('nfft', 384)
    window_type = params.get('window_type', 'blackman')
    
    hop_length = nperseg - noverlap
    
    # Convert to torch tensor
    force_cpu = os.environ.get("CAPC_FORCE_CPU") == "1"
    device = torch.device('cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    i_signal_torch = torch.from_numpy(i_component).float().to(device)
    
    # Create window
    if window_type == 'blackman':
        window = torch.blackman_window(nperseg, dtype=torch.float32, device=device)
    elif window_type == 'hann':
        window = torch.hann_window(nperseg, dtype=torch.float32, device=device)
    elif window_type == 'hamming':
        window = torch.hamming_window(nperseg, dtype=torch.float32, device=device)
    else:
        window = torch.ones(nperseg, dtype=torch.float32, device=device)
    
    # Compute STFT
    with torch.no_grad():
        spec_torch = torch.stft(
            i_signal_torch,
            n_fft=nfft,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        spec = torch.abs(spec_torch).cpu().numpy()
        
        # Take first 32 bins (as per original logic)
        spec = spec[:32, :]
    
    # Resize to square
    target_size = image_size
    
    if spec.shape[0] != target_size or spec.shape[1] != target_size:
        if spec.shape[0] != target_size:
            spec = signal.resample(spec, target_size, axis=0)
        if spec.shape[1] != target_size:
            spec = signal.resample(spec, target_size, axis=1)
    
    # Normalize
    spec_min, spec_max = spec.min(), spec.max()
    if spec_max > spec_min:
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    else:
        spec = np.zeros_like(spec)
        
    spec = spec[np.newaxis, :, :]
    
    return spec.astype(np.float32)

def create_iq_square_image(
    iq_data: np.ndarray,
    image_size: int = 224,
    last_n: Optional[int] = None,
    normalize: bool = True,
    method: str = 'magnitude',
) -> np.ndarray:
    """
    Convert the last N I/Q samples into a square grayscale image of shape (1, image_size, image_size).
    """
    # Handle different input shapes
    if iq_data.ndim != 2:
        raise ValueError(f"iq_data must be 2D, got shape {iq_data.shape}")
    
    # Convert to (2, N) format if needed
    if iq_data.shape[1] == 2:  # (N, 2) -> (2, N)
        iq_data = iq_data.T
    elif iq_data.shape[0] != 2:
        raise ValueError(f"iq_data must have 2 channels, got shape {iq_data.shape}")

    I = iq_data[0]
    Q = iq_data[1]

    if last_n is not None and last_n > 0:
        I = I[-last_n:]
        Q = Q[-last_n:]

    if method == 'magnitude':
        vec = np.sqrt(I.astype(np.float64) ** 2 + Q.astype(np.float64) ** 2)
    elif method == 'i':
        vec = I.astype(np.float64)
    elif method == 'q':
        vec = Q.astype(np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")

    target_len = int(image_size * image_size)
    cur_len = vec.shape[0]
    if cur_len == 0:
        img = np.zeros((image_size, image_size), dtype=np.float32)
        return img[np.newaxis, :, :]

    if cur_len == target_len:
        vec_resampled = vec
    elif cur_len > target_len:
        # Uniform downsampling by index selection
        idx = np.linspace(0, cur_len - 1, num=target_len, dtype=np.int64)
        vec_resampled = vec[idx]
    else:
        # Linear interpolation to upsample to target_len
        idx_src = np.arange(cur_len, dtype=np.float64)
        idx_tgt = np.linspace(0, cur_len - 1, num=target_len, dtype=np.float64)
        vec_resampled = np.interp(idx_tgt, idx_src, vec)

    img = vec_resampled.reshape(image_size, image_size)

    if normalize:
        vmin = img.min()
        vmax = img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)

    return img[np.newaxis, :, :].astype(np.float32)

def trim_leading_zeros(iq_data: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Trim leading near-zero samples from I/Q data."""
    magnitude = np.sqrt(iq_data[0]**2 + iq_data[1]**2)
    nonzero_idx = np.where(magnitude > threshold)[0]
    if len(nonzero_idx) > 0:
        start_idx = nonzero_idx[0]
        return iq_data[:, start_idx:]
    return iq_data

def expand_short_samples(iq_data: np.ndarray, min_length: int) -> np.ndarray:
    """Expand short samples via interpolation."""
    if iq_data.shape[1] >= min_length:
        return iq_data
        
    original_length = iq_data.shape[1]
    new_indices = np.linspace(0, original_length - 1, min_length)
    
    expanded_iq = np.zeros((2, min_length), dtype=iq_data.dtype)
    for i in range(2):  # I and Q channels
        expanded_iq[i] = np.interp(new_indices, np.arange(original_length), iq_data[i])
    
    return expanded_iq
