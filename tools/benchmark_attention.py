#!/usr/bin/env python3
"""Compare readable manual causal attention with PyTorch SDPA."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from build_llm.nn.attention import CausalSelfAttention


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(module, x, iterations: int) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(5):
            module(x)
        synchronize(x.device)
        start = time.perf_counter()
        for _ in range(iterations):
            module(x)
        synchronize(x.device)
    return (time.perf_counter() - start) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    manual = CausalSelfAttention(
        args.d_model,
        args.heads,
        args.seq_len,
        dropout=0.0,
        backend="manual",
    ).to(device)
    x = torch.randn(
        args.batch,
        args.seq_len,
        args.d_model,
        device=device,
    )
    print("device:", device)
    manual_ms = benchmark(manual, x, args.iterations)
    print(f"manual: {manual_ms:.3f} ms/forward")

    if not hasattr(F, "scaled_dot_product_attention"):
        print("SDPA unavailable in this PyTorch build")
        return

    sdpa = CausalSelfAttention(
        args.d_model,
        args.heads,
        args.seq_len,
        dropout=0.0,
        backend="sdpa",
    ).to(device)
    sdpa.load_state_dict(manual.state_dict())
    sdpa_ms = benchmark(sdpa, x, args.iterations)
    with torch.no_grad():
        max_error = float(
            (manual(x) - sdpa(x)).abs().max().item()
        )
    print(f"sdpa:   {sdpa_ms:.3f} ms/forward")
    print(f"speedup: {manual_ms / sdpa_ms:.2f}x")
    print(f"max absolute error: {max_error:.3e}")


if __name__ == "__main__":
    main()
