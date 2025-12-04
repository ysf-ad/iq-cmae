#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

# ----------------- Meta -------------------------------------------------------

def parse_sigmf_meta(meta_path: str) -> dict:
    with open(meta_path, "r") as f:
        meta = json.load(f)

    g = meta.get("global", {})
    emitters = g.get("ntia-emitter:emitters", [])
    emitter = emitters[0] if emitters else {}
    wf = emitter.get("waveform", {})
    anno = next((a for a in meta.get("annotations", []) if "sps" in a), {})

    return {
        "sps": int(anno.get("sps", 1)),
        "span": int(anno.get("span", 10)),
        "beta": float(anno.get("beta", 0.25)),
        "pilot_bits": wf.get("pilot_bits", 0),
        "modulation": wf.get("modulation", "unknown"),
        "fs": float(g.get("core:sample_rate", 1.0)),
        "f_if": float(emitter.get("intermediate_frequency", 0.0)), # Default to 0 if not found
        "datatype": g.get("core:datatype", "unknown"),
    }

# ----------------- Core Extraction Pipeline ------------------------------------

def extract_iq_data(data_path: str, meta_path: str, equalize: bool = True) -> np.ndarray:
    """
    Extract I/Q data from SIGMF files using Hilbert transform.
    
    Returns:
        full_iq_array: Full I/Q stream as (N, 2) array
    """
    if not Path(data_path).is_file():
        raise FileNotFoundError(data_path)
    if not Path(meta_path).is_file():
        raise FileNotFoundError(meta_path)

    meta = parse_sigmf_meta(meta_path)
    datatype = meta.get("datatype", "unknown")

    if datatype == 'ci16_le':
        # Complex Int16 Little Endian: [I, Q, I, Q, ...]
        raw = np.fromfile(data_path, dtype=np.int16)
        if raw.size == 0:
            raise ValueError("Empty/corrupt data")
        # Reshape to (N, 2)
        # Check if length is even
        if raw.size % 2 != 0:
            raw = raw[:-1]
        iq_array = raw.reshape(-1, 2).astype(np.float32)
        # Normalize to [-1, 1] range roughly? Or just keep as is. 
        # Usually int16 is -32768 to 32767. 
        # Let's normalize by 32768.0 to keep it in reasonable float range
        iq_array /= 32768.0
        return iq_array

    elif datatype == 'cf32_le':
        # Complex Float32 Little Endian: [I, Q, I, Q, ...]
        raw = np.fromfile(data_path, dtype=np.float32)
        if raw.size == 0:
            raise ValueError("Empty/corrupt data")
        if raw.size % 2 != 0:
            raw = raw[:-1]
        iq_array = raw.reshape(-1, 2)
        return iq_array

    else:
        # Assume Real-valued IF (float32) -> Use Hilbert
        # This matches original behavior for NE-Data (presumably)
        raw_if = np.fromfile(data_path, dtype=np.float32)
        
        if raw_if.size == 0:
            raise ValueError("Empty/corrupt data")

        # Do Hilbert transform once
        x_a = hilbert(raw_if)
        
        # Return full I/Q stream using the same analytic signal
        n = np.arange(len(raw_if))
        bb_full = x_a * np.exp(-1j*2*np.pi*meta["f_if"]*n/meta["fs"])
        iq_array = np.column_stack([np.real(bb_full), np.imag(bb_full)])

        return iq_array
 
