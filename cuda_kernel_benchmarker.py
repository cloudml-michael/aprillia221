"""
CUDA Kernel Benchmarker — profiles and compares custom CUDA kernel
performance against PyTorch baselines across batch sizes and precisions.
Built for GPU compute optimization on H100/H200 clusters.
"""

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class BenchmarkConfig:
    name: str
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32, 128, 512])
    seq_lengths: list[int] = field(default_factory=lambda: [128, 512, 2048])
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32, torch.float16])
    warmup_iters: int = 20
    benchmark_iters: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class KernelResult:
    kernel_name: str
    batch_size: int
    seq_length: int
    dtype: str
    mean_ms: float
    p99_ms: float
    tflops: float
    memory_mb: float


@dataclass
class BenchmarkReport:
    config_name: str
    results: list[KernelResult]
    baseline_results: list[KernelResult]

    def speedup_summary(self) -> dict:
        summary = {}
        baseline_map = {
            (r.batch_size, r.seq_length, r.dtype): r.mean_ms
            for r in self.baseline_results
        }
        for r in self.results:
            key = (r.batch_size, r.seq_length, r.dtype)
            if key in baseline_map:
                speedup = baseline_map[key] / max(r.mean_ms, 1e-6)
                summary[key] = round(speedup, 3)
        return summary


class CUDAKernelBenchmarker:
    """
    Benchmarks custom CUDA kernels vs PyTorch baselines.
    Measures latency, throughput (TFLOPS), and memory footprint.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)

    def _time_fn(self, fn: Callable, *args) -> tuple[float, float]:
        """Returns (mean_ms, p99_ms) over benchmark iterations."""
        # Warmup
        for _ in range(self.config.warmup_iters):
            fn(*args)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(self.config.benchmark_iters):
            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(*args)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                fn(*args)
                latencies.append((time.perf_counter() - t0) * 1000)

        latencies_sorted = sorted(latencies)
        mean_ms = sum(latencies) / len(latencies)
        p99_ms = latencies_sorted[int(0.99 * len(latencies))]
        return round(mean_ms, 4), round(p99_ms, 4)

    def _memory_mb(self) -> float:
        if self.device.type == "cuda":
            return round(torch.cuda.memory_allocated(self.device) / 1e6, 2)
        return 0.0

    def benchmark_matmul(self) -> list[KernelResult]:
        results = []
        for bs in self.config.batch_sizes:
            for dtype in self.config.dtypes:
                a = torch.randn(bs, 1024, device=self.device, dtype=dtype)
                b = torch.randn(1024, 1024, device=self.device, dtype=dtype)
                mean_ms, p99_ms = self._time_fn(torch.matmul, a, b)
                flops = 2 * bs * 1024 * 1024 / (mean_ms * 1e-3) / 1e12
                results.append(KernelResult(
                    kernel_name="matmul",
                    batch_size=bs,
                    seq_length=1024,
                    dtype=str(dtype),
                    mean_ms=mean_ms,
                    p99_ms=p99_ms,
                    tflops=round(flops, 3),
                    memory_mb=self._memory_mb(),
                ))
        return results

    def benchmark_attention(self) -> list[KernelResult]:
        results = []
        for bs in self.config.batch_sizes:
            for seq_len in self.config.seq_lengths:
                for dtype in self.config.dtypes:
                    q = torch.randn(bs, 8, seq_len, 64, device=self.device, dtype=dtype)
                    k = torch.randn(bs, 8, seq_len, 64, device=self.device, dtype=dtype)
                    v = torch.randn(bs, 8, seq_len, 64, device=self.device, dtype=dtype)
                    mean_ms, p99_ms = self._time_fn(
                        lambda: F.scaled_dot_product_attention(q, k, v)
                    )
                    results.append(KernelResult(
                        kernel_name="scaled_dot_product_attention",
                        batch_size=bs,
                        seq_length=seq_len,
                        dtype=str(dtype),
                        mean_ms=mean_ms,
                        p99_ms=p99_ms,
                        tflops=0.0,
                        memory_mb=self._memory_mb(),
                    ))
        return results
