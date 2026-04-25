#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path

import cupy as cp  # type: ignore
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "iq_cmae" / "data"))
sys.path.insert(0, str(ROOT / "iq_cmae" / "utils"))

from transforms import create_multimodal_image_from_torch  # type: ignore  # noqa: E402
from iq_extractor import extract_iq_data  # type: ignore  # noqa: E402


class PreprocessModule(nn.Module):
    def __init__(self, image_size: int, spectrogram_params: dict, gaf_clip_range: float = 3.0):
        super().__init__()
        self.image_size = image_size
        self.spectrogram_params = dict(spectrogram_params)
        self.gaf_clip_range = float(gaf_clip_range)

    def forward(self, iq_torch: torch.Tensor) -> torch.Tensor:
        return create_multimodal_image_from_torch(
            iq_torch,
            image_size=self.image_size,
            spectrogram_params=self.spectrogram_params,
            gaf_clip_range=self.gaf_clip_range,
            include_constellation=True,
            include_gaf=True,
            include_spectrogram=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark CUDA graph preprocessing from raw IQ to multimodal image")
    parser.add_argument("--data_root", type=str, default=str(ROOT / "data" / "ne-data" / "5 GHz Bandwidth"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--timing_iters", type=int, default=80)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def load_samples(base: Path, device: str) -> list[torch.Tensor]:
    samples: list[torch.Tensor] = []
    for class_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        file_path = next(class_dir.glob("frame_1.sigmf-data"))
        iq = extract_iq_data(str(file_path), str(file_path.with_suffix(".sigmf-meta")))
        iq = iq.T if iq.shape[1] == 2 else iq
        samples.append(torch.from_numpy(iq).to(device=device, dtype=torch.float32))
    return samples


def sync() -> None:
    torch.cuda.synchronize()
    cp.cuda.Stream.null.synchronize()


def bench_eager(module: nn.Module, samples: list[torch.Tensor], reps: int, warmup_iters: int) -> float:
    for _ in range(warmup_iters):
        for iq in samples:
            module(iq)
    sync()
    start = time.perf_counter()
    for _ in range(reps):
        for iq in samples:
            module(iq)
    sync()
    return (time.perf_counter() - start) * 1000.0 / (reps * len(samples))


def bench_cuda_graph(module: nn.Module, samples: list[torch.Tensor], reps: int, warmup_iters: int) -> float:
    static_input = samples[0].clone()
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(warmup_iters):
            static_output = module(static_input)
    torch.cuda.current_stream().wait_stream(warm_stream)
    sync()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = module(static_input)

    for _ in range(10):
        for iq in samples:
            static_input.copy_(iq)
            graph.replay()
    sync()

    start = time.perf_counter()
    for _ in range(reps):
        for iq in samples:
            static_input.copy_(iq)
            graph.replay()
    sync()
    return (time.perf_counter() - start) * 1000.0 / (reps * len(samples))


def main() -> None:
    args = parse_args()
    os.environ["CAPC_GPU_CONSTELLATION"] = "1"
    os.environ["CAPC_GPU_GAF"] = "1"
    os.environ["CAPC_TORCH_FFT_RESAMPLE"] = "1"
    os.environ.pop("CAPC_CUPY_RESAMPLE", None)

    output_path = Path(args.output) if args.output else ROOT / "outputs" / "cuda_graph_preprocess_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(Path(args.data_root), args.device)
    params = {"nperseg": 192, "noverlap": 144, "nfft": 384, "window_type": "blackman"}
    module = PreprocessModule(image_size=args.image_size, spectrogram_params=params).to(args.device).eval()

    eager_ms = bench_eager(module, samples, reps=args.timing_iters, warmup_iters=args.warmup_iters)
    graph_ms = bench_cuda_graph(module, samples, reps=args.timing_iters, warmup_iters=args.warmup_iters)

    out = {
        "device": args.device,
        "samples": len(samples),
        "eager_ms_per_sample": eager_ms,
        "cuda_graph_ms_per_sample": graph_ms,
        "speedup": eager_ms / graph_ms,
    }
    output_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
