from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable

# Profilers are read lazily in the native extension, but set these before
# importing the package so a coverage run is self-contained.
os.environ.setdefault("XDNA_PROFILE_FALLBACK", "1")
os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

import xdna_train


TensorArgs = tuple[torch.Tensor, ...]


@dataclass
class Motif:
    name: str
    model_factory: Callable[[], nn.Module]
    input_factory: Callable[[], tuple[TensorArgs, torch.Tensor | None]]
    loss_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
    description: str
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class UNetMotif(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Conv2d(16, 32, 3, padding=1)
        self.dec = nn.Conv2d(48, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(self.enc(x), 0.1)
        y = F.interpolate(y, scale_factor=2.0, mode="nearest")
        skip = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.dec(torch.cat((y, skip), dim=1))


class NisMiniConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class NisMiniResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(128)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x.clone()
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + skip)


class NisMiniUpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.act(self.norm(self.conv(x)))


class NisMiniNet(nn.Module):
    """Scaled spatial clone of NIS MyNet(3, 1, 7).

    Channel widths and block count intentionally match the real network. Only
    spatial dimensions are reduced by the input factory.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = NisMiniConvBlock(3, 64)
        self.conv1 = NisMiniConvBlock(64, 128, stride=2)
        self.conv2 = NisMiniConvBlock(128, 128, stride=2)
        self.resnet_blocks = nn.ModuleList(
            [NisMiniResidualBlock() for _ in range(7)]
        )
        self.upconv2 = NisMiniUpsampleConvBlock(256, 128)
        self.upconv1 = NisMiniUpsampleConvBlock(256, 128)
        self.conv11 = NisMiniConvBlock(195, 64)
        self.conv12 = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x_res = x2.clone()
        for block in self.resnet_blocks:
            x_res = block(x_res)
        x_out = self.upconv2(torch.cat((x_res, x2), dim=1))
        x_out = self.upconv1(torch.cat((x_out, x1), dim=1))
        x_out = self.conv11(torch.cat((x_out, x0, x), dim=1))
        return self.conv12(x_out)


class TransformerMotif(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(64)
        self.ff1 = nn.Linear(64, 128)
        self.ff2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        return self.norm2(x + self.ff2(F.gelu(self.ff1(x))))


class MobileMotif(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.pw = nn.Conv2d(32, 64, 1, bias=False)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.dw(x))
        x = F.silu(self.pw(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


class EmbeddingMotif(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(4096, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(token_ids).mean(dim=1)
        return self.fc2(F.gelu(self.fc1(x)))


def _bf16_randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).bfloat16()


def motifs() -> dict[str, Motif]:
    return {
        "nis-mini": Motif(
            "nis-mini",
            NisMiniNet,
            lambda: (
                (
                    _bf16_randn(4, 3, 72, 68).contiguous(
                        memory_format=torch.channels_last
                    ),
                ),
                torch.rand(4, 1, 72, 68).bfloat16().contiguous(
                    memory_format=torch.channels_last
                ),
            ),
            lambda out, target: F.binary_cross_entropy(torch.sigmoid(out), target),
            "Scaled NIS MyNet: same 7 blocks/channels/op graph at 72x68 + BCE + Adam",
            lambda model: torch.optim.Adam(
                model.parameters(), lr=1e-4, weight_decay=1e-5
            ),
        ),
        "mlp": Motif(
            "mlp",
            lambda: nn.Sequential(
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
            ),
            lambda: (((_bf16_randn(16, 32, 128)),), _bf16_randn(16, 32, 64)),
            lambda out, target: F.mse_loss(out, target),
            "Linear + GELU + LayerNorm + MSE + AdamW",
        ),
        "resnet": Motif(
            "resnet",
            ResidualBlock,
            lambda: (((_bf16_randn(4, 32, 32, 32).contiguous(
                memory_format=torch.channels_last)),), _bf16_randn(
                    4, 32, 32, 32
                ).contiguous(memory_format=torch.channels_last)),
            lambda out, target: F.mse_loss(out, target),
            "Conv2d + BatchNorm + residual add + ReLU",
        ),
        "unet": Motif(
            "unet",
            UNetMotif,
            lambda: (((_bf16_randn(2, 16, 32, 32).contiguous(
                memory_format=torch.channels_last)),), torch.rand(
                    2, 1, 64, 64
                ).bfloat16().contiguous(memory_format=torch.channels_last)),
            lambda out, target: F.binary_cross_entropy_with_logits(out, target),
            "Conv2d + nearest upsample + cat + BCE-with-logits",
        ),
        "transformer": Motif(
            "transformer",
            TransformerMotif,
            lambda: (((_bf16_randn(4, 32, 64)),), _bf16_randn(4, 32, 64)),
            lambda out, target: F.mse_loss(out, target),
            "MHA + LayerNorm + residual + GELU MLP",
        ),
        "mobile": Motif(
            "mobile",
            MobileMotif,
            lambda: (((_bf16_randn(4, 32, 32, 32).contiguous(
                memory_format=torch.channels_last)),), torch.randint(0, 10, (4,))),
            lambda out, target: F.cross_entropy(out, target),
            "Depthwise Conv + pointwise Conv + SiLU + pooling + CE",
        ),
        "embedding": Motif(
            "embedding",
            EmbeddingMotif,
            lambda: (((torch.randint(0, 4096, (16, 32), dtype=torch.int64)),),
                     torch.randint(0, 16, (16,), dtype=torch.int64)),
            lambda out, target: F.cross_entropy(out, target),
            "Embedding/gather + reduction + GELU MLP + CE",
        ),
    }


def _to_device(value: torch.Tensor, device: str) -> torch.Tensor:
    return value.to(device)


def _normalize_stats(
    stats: dict[str, dict[str, float | int]], steps: int
) -> list[dict[str, float | int | str]]:
    rows = []
    for name, item in stats.items():
        total = float(item["total_ms"])
        rows.append(
            {
                "op": name,
                "calls": int(item["calls"]),
                "total_ms": total,
                "ms_per_step": total / max(steps, 1),
            }
        )
    return sorted(rows, key=lambda row: float(row["total_ms"]), reverse=True)


def run_motif(
    motif: Motif,
    *,
    warmup: int,
    steps: int,
    threads: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = motif.model_factory().bfloat16().to("xdna")
    cpu_args, cpu_target = motif.input_factory()
    args = tuple(_to_device(x, "xdna") for x in cpu_args)
    target = None if cpu_target is None else _to_device(cpu_target, "xdna")
    optimizer = (
        motif.optimizer_factory(model)
        if motif.optimizer_factory is not None
        else torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False, fused=False
        )
    )

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        out = model(*args)
        loss = motif.loss_fn(out, target)
        loss.backward()
        optimizer.step()
        torch.xdna.synchronize()
        return loss

    for _ in range(warmup):
        step()

    xdna_train.reset_fallback_stats()
    xdna_train.reset_mapped_cpu_stats()
    times = []
    loss = None
    for _ in range(steps):
        start = time.perf_counter()
        loss = step()
        times.append((time.perf_counter() - start) * 1000.0)

    assert loss is not None
    loss_value = float(loss.detach().cpu())
    fallbacks = _normalize_stats(xdna_train.fallback_stats(), steps)
    mapped = _normalize_stats(xdna_train.mapped_cpu_stats(), steps)
    fallback_total = sum(float(x["total_ms"]) for x in fallbacks)
    mapped_total = sum(float(x["total_ms"]) for x in mapped)
    return {
        "motif": motif.name,
        "description": motif.description,
        "status": "ok",
        "steps": steps,
        "warmup": warmup,
        "wall_ms": times,
        "wall_median_ms": float(torch.tensor(times).median()),
        "wall_mean_ms": sum(times) / len(times),
        "loss": loss_value,
        "loss_finite": math.isfinite(loss_value),
        "generic_fallback_total_ms": fallback_total,
        "generic_fallback_ms_per_step": fallback_total / steps,
        "mapped_cpu_total_ms": mapped_total,
        "mapped_cpu_ms_per_step": mapped_total / steps,
        "generic_fallbacks": fallbacks,
        "mapped_cpu": mapped,
    }


def _print_result(result: dict, top: int) -> None:
    print(f"\n[{result['motif']}] {result.get('description', '')}")
    if result["status"] != "ok":
        print(f"  ERROR: {result['error']}")
        return
    print(
        f"  wall median={result['wall_median_ms']:.3f} ms "
        f"mean={result['wall_mean_ms']:.3f} ms "
        f"loss={result['loss']:.7g}"
    )
    print(
        f"  generic fallback={result['generic_fallback_ms_per_step']:.3f} ms/step "
        f"mapped-CPU={result['mapped_cpu_ms_per_step']:.3f} ms/step"
    )
    if result["generic_fallbacks"]:
        print("  top generic fallbacks:")
        for row in result["generic_fallbacks"][:top]:
            print(
                f"    {row['op']:<42} calls={row['calls']:<4} "
                f"{row['ms_per_step']:.3f} ms/step"
            )
    if result["mapped_cpu"]:
        print("  top mapped-CPU handlers:")
        for row in result["mapped_cpu"][:top]:
            print(
                f"    {row['op']:<42} calls={row['calls']:<4} "
                f"{row['ms_per_step']:.3f} ms/step"
            )


def main() -> None:
    available = motifs()
    ap = argparse.ArgumentParser(
        description="Training-motif coverage/profiling suite for torch-xdna2"
    )
    ap.add_argument(
        "--motif",
        action="append",
        choices=["all", *available.keys()],
        help="Motif to run. Repeat for several; default: all.",
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    requested = args.motif or ["all"]
    names = list(available) if "all" in requested else requested
    results = []
    for name in names:
        try:
            result = run_motif(
                available[name],
                warmup=args.warmup,
                steps=args.steps,
                threads=args.threads,
                seed=args.seed,
            )
        except Exception as exc:
            result = {
                "motif": name,
                "description": available[name].description,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        if not args.json:
            _print_result(result, args.top)

    if args.json:
        print(json.dumps(results, indent=2))
    elif any(r["status"] != "ok" for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
