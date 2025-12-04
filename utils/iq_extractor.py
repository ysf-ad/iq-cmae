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

    g = meta["global"]
    emitter = g["ntia-emitter:emitters"][0]
    wf = emitter["waveform"]
    anno = next((a for a in meta.get("annotations", []) if "sps" in a), {})

    return {
        "sps": int(anno.get("sps", 1)),
        "span": int(anno.get("span", 10)),
        "beta": float(anno.get("beta", 0.25)),
        "pilot_bits": wf["pilot_bits"],
        "modulation": wf["modulation"],
        "fs": float(g["core:sample_rate"]),
        "f_if": float(emitter.get("intermediate_frequency", 10e9)),
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
 
