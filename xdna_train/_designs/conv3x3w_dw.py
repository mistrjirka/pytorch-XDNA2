"""conv3x3w_dw: weight gradient of a 3x3 stride-1 pad-1 convolution on XDNA2.

dW[co][ci][dy][dx] = sum_p X[p + dy*(W+2) + dx][ci] * dY[p][co] over padded-grid
pixels p: bfp16 operands (8-pixel blocks) with fp32 sums (conv3x3w_dw_kernel.cc).

Inputs reuse the forward family's packed input layout (conv3x3w_host.pack_input):
  X:  pack_input(x,  chunks)   [groups][chunks][2126][64]
  dY: pack_input(dy, cochunks) [groups][cochunks][2126][64]; pixel p of dY sits at
      grid row p + (W+2) + 1, and every padding position reads as zero.

A dispatch ("pass") covers 2 input-channel chunks x 2 output-channel chunks
(128 x 128 channels, all 9 taps) on 6 x 4 cores:
  column c = 3*chunk_local + dy: its shim streams the (chunk, dy) windows of
    66 rows per 64-pixel step, broadcast to the column's 4 cores;
  row q: output channels [32q, 32q+32) of the pass, streamed by shim column q
    and broadcast across the 6 columns.
Each core accumulates its [3 dx][64 ci][32 co] tile over all pixels and drains
it once at the end:
  C: [6 cols][4 rows][3 dx][8 ci blk][4 co blk][8 ci][8 co] fp32.
Channel chunks beyond the tensor are read from chunk 0 (results discarded) so
every pass runs the same core program; only the instruction stream differs.

Build: python conv3x3w_dw.py --out DIR --groups G --wp W+2 --chunks C --cochunks D
       --ci-pass P --co-pass Q   (writes DIR/conv3x3w_dw.xclbin + DIR/<key>.bin)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program,
                      Runtime, Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

KERNEL_SRC = Path(__file__).resolve().parent / "conv3x3w_dw_kernel.cc"

COLS, ROWS = 6, 4
STEP = 64            # pixels per core step
WIN = STEP + 2       # window rows per (dy, step): 3 dx taps
K_STEP = 64          # channels per chunk
CO_ROW = 32          # output channels per core row
GROUP_OUT_ROWS = 2048
GROUP_ROWS = 2126
STEPS_PER_GROUP = GROUP_OUT_ROWS // STEP
ACC = 3 * 64 * CO_ROW  # fp32 per core
MAX_ITER = 64        # shim BD iteration counter limit
RTP_LEN = 4


def stream_key(groups, wp, chunks, cochunks, ci_pass, co_pass):
    return f"g{groups}_w{wp}_c{chunks}_d{cochunks}_p{ci_pass}_q{co_pass}"


def default_aie_kernels():
    import aie
    return str(Path(aie.__file__).resolve().parents[2] / "include" / "aie_kernels")


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def conv3x3w_dw(
    X: In,
    D: In,
    C: Out,
    *,
    groups: CompileTime[int] = 29,
    wp: CompileTime[int] = 38,
    chunks: CompileTime[int] = 4,
    cochunks: CompileTime[int] = 2,
    ci_pass: CompileTime[int] = 0,
    co_pass: CompileTime[int] = 0,
    aie_kernels: CompileTime[str] = "",
    debug_flags: CompileTime[str] = "",
):
    assert 3 <= wp <= 38 and groups >= 1
    # Shim strides are limited to 2^20 32-bit words.
    assert max(chunks, cochunks) * GROUP_ROWS * K_STEP // 2 < (1 << 20)

    X_l2_ty = np.ndarray[(WIN, K_STEP), np.dtype[bfloat16]]
    D_l2_ty = np.ndarray[(STEP, CO_ROW), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(ACC,), np.dtype[np.float32]]
    C_l2_ty = np.ndarray[(ROWS * ACC,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    dyb_ty = np.ndarray[(4 * STEP,), np.dtype[v8bfp16ebs8]]  # 32 converted dY blocks

    flags = [f"-I{aie_kernels}", f"-DWIN_R={WIN}"] + debug_flags.split()
    zero = ExternalFunction("zero_dw_f32", source_file=str(KERNEL_SRC), arg_types=[acc_ty],
                            compile_flags=flags + ["-DZERO_ONLY"], use_chess=False)
    dw = ExternalFunction("dw_window", source_file=str(KERNEL_SRC),
                          arg_types=[X_l2_ty, D_l2_ty, acc_ty, dyb_ty],
                          compile_flags=flags + ["-DDW_ONLY"], use_chess=False)

    # Rows arrive row-major; store 8-channel blocked so 8x8 blocks are contiguous.
    x_l1_dims = [(WIN, 8), (K_STEP // 8, WIN * 8), (8, 1)]
    d_l1_dims = [(STEP, 8), (CO_ROW // 8, STEP * 8), (8, 1)]

    X_l3 = [ObjectFifo(X_l2_ty, name=f"X_L3L2_{c}", depth=2) for c in range(COLS)]
    X_l1 = [X_l3[c].cons().forward(tile=Tile(c, 1), obj_type=X_l2_ty, name=f"X_L2L1_{c}",
                                   depth=2) for c in range(COLS)]
    D_l3 = [ObjectFifo(D_l2_ty, name=f"D_L3L2_{q}", depth=2) for q in range(ROWS)]
    D_l1 = [D_l3[q].cons().forward(tile=Tile(q, 1), obj_type=D_l2_ty, name=f"D_L2L1_{q}",
                                   depth=2) for q in range(ROWS)]
    C_l2 = [ObjectFifo(C_l2_ty, name=f"C_L2L3_{c}", depth=1) for c in range(COLS)]
    C_l1 = [C_l2[c].prod().join([ACC * q for q in range(ROWS)], tile=Tile(c, 1),
                                obj_types=[acc_ty] * ROWS,
                                names=[f"C_L1L2_{c}_{q}" for q in range(ROWS)],
                                depths=[1] * ROWS) for c in range(COLS)]

    rtps = [[Buffer(rtp_ty, name=f"rtp_{c}_{q}", initial_value=np.zeros(RTP_LEN, np.int32),
                    use_write_rtp=True) for q in range(ROWS)] for c in range(COLS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(ROWS)] for _ in range(COLS)]
    scratch = [[Buffer(dyb_ty, name=f"dyb_{c}_{q}") for q in range(ROWS)] for c in range(COLS)]

    def core_fn(in_x, in_d, out_c, zero_k, dw_k, rtp, barrier, dyb):
        # wait_for_value is an acquire-equal that leaves the lock set; release
        # so the next dispatch waits for its own parameters.
        barrier.wait_for_value(1)
        n_steps = rtp[0]
        barrier.release_with_value(1)
        acc = out_c.acquire(1)
        zero_k(acc)
        for _ in range_(n_steps):
            xw = in_x.acquire(1)
            dd = in_d.acquire(1)
            dw_k(xw, dd, acc, dyb)
            in_x.release(1)
            in_d.release(1)
        out_c.release(1)

    workers = [Worker(core_fn,
                      [X_l1[c].cons(dims_from_stream=x_l1_dims),
                       D_l1[q].cons(dims_from_stream=d_l1_dims),
                       C_l1[c][q].prod(), zero, dw, rtps[c][q], barriers[c][q],
                       scratch[c][q]],
                      tile=Tile(c, 2 + q), stack_size=0xC00)
               for c in range(COLS) for q in range(ROWS)]

    x_elems = groups * chunks * GROUP_ROWS * K_STEP
    d_elems = groups * cochunks * GROUP_ROWS * K_STEP
    X_ty = np.ndarray[(x_elems,), np.dtype[bfloat16]]
    D_ty = np.ndarray[(d_elems,), np.dtype[bfloat16]]
    C_ty = np.ndarray[(COLS * ROWS * ACC,), np.dtype[np.float32]]

    def x_tap(col, g0, ng):
        chunk = 2 * ci_pass + col // 3
        chunk = chunk if chunk < chunks else 0
        dy = col % 3
        base = (g0 * chunks + chunk) * GROUP_ROWS * K_STEP + dy * wp * K_STEP
        return TensorAccessPattern((x_elems,), base, [ng, STEPS_PER_GROUP, WIN, K_STEP],
                                   [chunks * GROUP_ROWS * K_STEP, STEP * K_STEP, K_STEP, 1])

    def d_tap(q, g0, ng):
        chunk = 2 * co_pass + q // 2
        chunk = chunk if chunk < cochunks else 0
        base = ((g0 * cochunks + chunk) * GROUP_ROWS + wp + 1) * K_STEP + (q % 2) * CO_ROW
        return TensorAccessPattern((d_elems,), base, [ng, STEPS_PER_GROUP, STEP, CO_ROW],
                                   [cochunks * GROUP_ROWS * K_STEP, STEP * K_STEP, K_STEP, 1])

    def write_params(*bufs):
        for buf in bufs:
            buf[0] = groups * STEPS_PER_GROUP

    rt = Runtime()
    with rt.sequence(X_ty, D_ty, C_ty) as (x, d, cc):
        rt.start(*workers)
        rt.inline_ops(write_params, [b for col in rtps for b in col])
        for col in barriers:
            for bar in col:
                rt.set_barrier(bar, 1)
        # One fill task per channel per <= 64 groups (iteration counter limit);
        # at most 2 in flight so a shim task queue (depth 4) never overflows.
        prev = None
        for g0 in range(0, groups, MAX_ITER):
            ng = min(MAX_ITER, groups - g0)
            tg = rt.task_group()
            for c in range(COLS):
                rt.fill(X_l3[c].prod(), x, tap=x_tap(c, g0, ng), task_group=tg, wait=True,
                        tile=Tile(c, 0))
            for q in range(ROWS):
                rt.fill(D_l3[q].prod(), d, tap=d_tap(q, g0, ng), task_group=tg, wait=True,
                        tile=Tile(q, 0))
            if prev is not None:
                rt.finish_task_group(prev)
            prev = tg
        tg = rt.task_group()
        for c in range(COLS):
            tap = TensorAccessPattern((COLS * ROWS * ACC,), c * ROWS * ACC, [1, 1, 1, ROWS * ACC],
                                      [0, 0, 0, 1])
            rt.drain(C_l2[c].cons(), cc, tap=tap, task_group=tg, wait=True, tile=Tile(c, 0))
        rt.finish_task_group(prev)
        rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()


def build(out_dir, groups, wp, chunks, cochunks, ci_pass, co_pass, aie_kernels=None,
          keep_prj=False, debug_flags=""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = stream_key(groups, wp, chunks, cochunks, ci_pass, co_pass)
    tmp_x = out / f".{key}.xclbin.tmp"
    tmp_i = out / f".{key}.bin.tmp"
    from aie.iron.device import from_name
    from aie.utils.hostruntime import set_current_device
    set_current_device(from_name("npu2", n_cols=None))
    conv3x3w_dw.specialize(groups=groups, wp=wp, chunks=chunks, cochunks=cochunks,
                           ci_pass=ci_pass, co_pass=co_pass,
                           aie_kernels=aie_kernels or default_aie_kernels(),
                           debug_flags=debug_flags).compile(
        xclbin_path=str(tmp_x), inst_path=str(tmp_i))
    shared = out / "conv3x3w_dw.xclbin"
    if not shared.exists():
        tmp_x.replace(shared)
    else:
        tmp_x.unlink()
    if not keep_prj:
        shutil.rmtree(out / f".{key}.xclbin.prj", ignore_errors=True)
    tmp_i.replace(out / f"{key}.bin")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for name in ("groups", "wp", "chunks", "cochunks", "ci-pass", "co-pass"):
        ap.add_argument(f"--{name}", type=int, required=True)
    ap.add_argument("--aie-kernels", default=None)
    ap.add_argument("--keep-prj", action="store_true")
    ap.add_argument("--debug-flags", default="")
    a = ap.parse_args()
    print(build(a.out, a.groups, a.wp, a.chunks, a.cochunks, a.ci_pass, a.co_pass,
                a.aie_kernels, a.keep_prj, a.debug_flags))


if __name__ == "__main__":
    main()
