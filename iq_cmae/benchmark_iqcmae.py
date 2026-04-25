import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile

from models.iqcmae_model import IQCMAE


class EncoderOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        latent, _, _ = self.model.forward_encoder(x, mask_ratio=0.0)
        return latent[:, 1:, :].mean(dim=1)


def load_checkpoint_if_present(model, checkpoint_path, device):
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_target_missing = [k for k in missing if not k.startswith("target_")]
    if non_target_missing or unexpected:
        raise ValueError(
            "Checkpoint is not compatible with IQCMAE benchmark. "
            f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )
    if missing and hasattr(model, "_init_target_network"):
        model._init_target_network()


def benchmark_eager(module, sample, device, warmup_iters, timing_iters):
    if device.type == "cuda":
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = module(sample)
        torch.cuda.synchronize()
    else:
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = module(sample)

    start_time = time.perf_counter()
    for _ in range(timing_iters):
        with torch.no_grad():
            _ = module(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time_s = time.perf_counter() - start_time
    avg_latency_ms = (total_time_s / timing_iters) * 1000.0
    throughput = timing_iters / total_time_s if total_time_s > 0 else 0.0
    return float(avg_latency_ms), float(throughput)


def benchmark_cuda_graph(module, sample, warmup_iters, timing_iters):
    if not sample.is_cuda:
        raise ValueError("CUDA-graph benchmarking requires a CUDA device")

    static_input = sample.clone()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = module(static_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = module(static_input)

    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(timing_iters):
        graph.replay()
    torch.cuda.synchronize()
    total_time_s = time.perf_counter() - start_time
    avg_latency_ms = (total_time_s / timing_iters) * 1000.0
    throughput = timing_iters / total_time_s if total_time_s > 0 else 0.0
    return float(avg_latency_ms), float(throughput), static_output.shape


def main(args):
    device = torch.device(args.device)

    model = IQCMAE(
        img_size=args.input_size,
        patch_size=16,
        in_chans=6,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        contrastive_weight=args.cw,
        contrastive_last_k=args.k,
        shared_layers=args.s,
        fusion_type=args.fusion_type,
    ).to(device)
    model.eval()
    load_checkpoint_if_present(model, args.checkpoint, device)

    wrapper = EncoderOnlyWrapper(model).to(device)
    wrapper.eval()

    params = sum(p.numel() for p in wrapper.parameters())
    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)

    sample = torch.randn(1, 6, args.input_size, args.input_size, device=device)

    with torch.no_grad():
        macs, _ = profile(wrapper, inputs=(sample,), verbose=False)

    eager_latency_ms, eager_throughput = benchmark_eager(
        wrapper, sample, device, args.warmup_iters, args.timing_iters
    )

    cuda_graph_latency_ms = None
    cuda_graph_throughput = None
    if device.type == "cuda" and args.cuda_graph:
        cuda_graph_latency_ms, cuda_graph_throughput, _ = benchmark_cuda_graph(
            wrapper, sample, args.warmup_iters, args.timing_iters
        )

    results = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "input_shape": [1, 6, args.input_size, args.input_size],
        "shared_layers": args.s,
        "contrastive_last_k": args.k,
        "contrastive_weight": args.cw,
        "parameters": int(params),
        "trainable_parameters": int(trainable_params),
        "macs": float(macs),
        "flops_estimate": float(macs * 2.0),
        "warmup_iters": args.warmup_iters,
        "timing_iters": args.timing_iters,
        "eager_latency_ms": eager_latency_ms,
        "eager_throughput_samples_per_sec": eager_throughput,
        "cuda_graph_enabled": bool(device.type == "cuda" and args.cuda_graph),
        "cuda_graph_latency_ms": cuda_graph_latency_ms,
        "cuda_graph_throughput_samples_per_sec": cuda_graph_throughput,
        "cuda_graph_speedup": (
            eager_latency_ms / cuda_graph_latency_ms
            if cuda_graph_latency_ms and cuda_graph_latency_ms > 0
            else None
        ),
    }

    print(json.dumps(results, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark IQ-CMAE encoder inference")
    parser.add_argument("--checkpoint", default="", type=str, help="Optional checkpoint path")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--embed_dim", default=192, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num_heads", default=3, type=int)
    parser.add_argument("--s", default=9, type=int, help="Shared layers")
    parser.add_argument("--k", default=4, type=int, help="Contrastive last k layers")
    parser.add_argument("--cw", default=2.5, type=float, help="Contrastive weight")
    parser.add_argument("--fusion_type", default="concat", type=str)
    parser.add_argument("--warmup_iters", default=20, type=int)
    parser.add_argument("--timing_iters", default=100, type=int)
    parser.add_argument("--cuda_graph", action="store_true", help="Also benchmark CUDA graph replay latency")
    parser.add_argument("--output", default="", type=str, help="Optional JSON output path")
    main(parser.parse_args())
