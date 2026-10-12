#!/usr/bin/env python3
"""Regression: discarded lazy cat must preserve a recycled BO's dirty state.

Runs the default (lazy disabled) control and opt-in lazy case in separate,
serial processes. Neither reads/materializes the cat before deleting it.
Each case requires the poisoned allocation to be reused by cat and NPU GEMM,
and checks the GEMM result against a CPU reference. A regression exits nonzero.

Run with the hardware Python and PYTHONPATH pointing at this checkout:
    python tests/run_lazy_recycle_coherency.py
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def run_case(case):
    import torch
    import xdna_train
    from xdna_train import device

    xdna_train.register_xdna_device()
    xdna_train.set_npu_mode(2)
    failures = []

    def check(name, ok, detail=""):
        print(f"{name}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
        if not ok:
            failures.append(name)

    print(f"case={case} package={xdna_train.__file__}", flush=True)
    print(f"native={xdna_train.native_library_path()}", flush=True)
    print("expected: pointer reuse, NPU GEMM, max error < 0.01, zero elements equal to 42",
          flush=True)
    check("GEMM family configured", device._C.gemm_family_configured())
    if failures:
        return 1

    torch.manual_seed(42)
    a = (torch.randn(256, 768) * .03).bfloat16()
    b = (torch.randn(768, 3072) * .03).bfloat16()
    ref = (a.float() @ b.float()).bfloat16().float()
    ax, bx = a.to("xdna"), b.to("xdna")
    # cat output has exactly the same bytes as the GEMM output.
    left = torch.ones(1, 32, 96, 128, dtype=torch.bfloat16).to(
        device="xdna", memory_format=torch.channels_last)
    right = torch.full((1, 32, 96, 128), 2., dtype=torch.bfloat16).to(
        device="xdna", memory_format=torch.channels_last)

    xdna_train.reset_mapped_cpu_stats()
    reused = 0
    worst_error = 0.0
    total_poison = 0
    for attempt in range(1, 4):
        xdna_train.empty_cache()
        poison = torch.full((256, 3072), 42., device="xdna", dtype=torch.bfloat16)
        poison_ptr = poison.data_ptr()
        del poison

        cat = torch.cat((left, right), dim=1)
        cat_ptr = cat.data_ptr()
        # Metadata inspection only: reading cat would materialize it and hide
        # the bug in set_virtual_value's handling of the recycled dirty state.
        cat_state = xdna_train.coherency_state(cat)
        del cat

        out = ax @ bx
        out_ptr = out.data_ptr()
        same = poison_ptr == cat_ptr == out_ptr
        reused += int(same)
        got = out.cpu().float()
        error = (got - ref).abs().max().item()
        poison_count = int((got == 42).sum().item())
        worst_error = max(worst_error, error)
        total_poison += poison_count
        print(
            f"attempt={attempt} reused={same} "
            f"ptrs={poison_ptr:#x}/{cat_ptr:#x}/{out_ptr:#x} "
            f"cat_coherency={cat_state} max_error={error:.9f} "
            f"poison_42={poison_count}/{got.numel()}",
            flush=True,
        )
        check(f"attempt {attempt}: poison -> cat -> GEMM reused allocation", same)
        check(f"attempt {attempt}: correct GEMM",
              error < .01 and poison_count == 0)
        del out, got

    fallback_calls = xdna_train.mapped_cpu_stats().get(
        "aten::mm.cpu_mapped", {}).get("calls", 0)
    check("all GEMMs ran on NPU", fallback_calls == 0,
          f"CPU mm fallback calls={fallback_calls}")
    check("all attempts exercised recycled allocation", reused == 3,
          f"reused={reused}/3")
    print(
        f"case={case} {'PASS' if not failures else 'FAIL'} "
        f"reused={reused}/3 max_error={worst_error:.9f} "
        f"total_poison_42={total_poison} failures={len(failures)}",
        flush=True,
    )
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("default-off", "lazy-on"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        # Set before importing torch/backend, including for standalone cases.
        os.environ["XDNA_PROFILE_MAPPED_CPU"] = "1"
        if args.case == "default-off":
            os.environ.pop("XDNA_LAZY_DECODER_VALUES", None)
        else:
            os.environ["XDNA_LAZY_DECODER_VALUES"] = "1"
        return run_case(args.case)

    results = {}
    for case in ("default-off", "lazy-on"):
        print(f"\n=== {case} ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--case", case],
            check=False,
        )
        results[case] = result.returncode
    print(f"case exit codes: {results}", flush=True)
    print("ALL OK" if all(code == 0 for code in results.values()) else "FAILED",
          flush=True)
    return int(any(code != 0 for code in results.values()))


if __name__ == "__main__":
    sys.exit(main())
