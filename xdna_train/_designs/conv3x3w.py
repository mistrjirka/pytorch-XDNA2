"""conv3x3w: 3x3 stride-1 pad-1 convolution family on the full XDNA2 array.

One core program serves every shape up to the family limits; a shape only
changes the instruction stream.  Each dispatch writes, per core, a runtime
parameter block [groups, channel_chunks, 9 tap row offsets] and releases a
barrier, so switching layers costs no hardware-context switch.

Data flow (bf16 activations x bfp16ebs8 weights -> bf16, BF16 running sum):
  * Output rows are padded-grid pixels: row p of image b is the window whose
    top-left input pixel is p, so tap (dy, dx) is row offset dy*(W+2)+dx.
  * Each of the 32 cores owns M = 64 consecutive output rows and all N = 128
    output channels.  Per (group, 64-channel chunk) a column's shim streams its
    4 cores' input windows (64 + halo rows x 64 channels) through the memtile;
    each core keeps its window in L1 and reads the 9 taps by pointer offset.
  * Weights (K ordered (chunk, dy, dx, c)) are read once per group and
    broadcast to all cores.

Two hardware rules this design depends on (each one hung the array):
  * WorkerRuntimeBarrier.wait_for_value is an acquire-equal that leaves the
    lock set; the core must release_with_value afterwards or every later
    dispatch runs with the previous dispatch's runtime parameters.
  * A shim DMA channel queues at most 4 tasks; a 5th push is dropped.  Each
    group therefore issues one task per channel and at most 4 are in flight.

Host layouts (see xdna_train conv packing):
  X: [groups][chunks][GROUP_ROWS][64] bf16, GROUP_ROWS = 2048 + WIN_ROWS - 64,
     padded NHWC rows with each group's halo duplicated
  B: [K/64 * N/8 ... ] bfp16ebs8 in AMD's ATB 1x2/8x8 shuffle, K = 576 * chunks
  C: [groups * 2048][128] bf16 padded-grid rows

Build (mlir-aie IRON environment):
  python conv3x3w.py --out DIR --groups G --wp W+2 --chunks C
  (or --batch B --height H --width W instead of --groups/--wp)
writes DIR/conv3x3w.xclbin (shape independent) and DIR/<stream key>.bin.
A stream depends only on (groups, padded width, channel chunks), so shapes
with the same three values share it.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program,
                      Runtime, Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

KERNEL_SRC = Path(__file__).resolve().parent / "conv3x3w_kernel.cc"

# Family constants (the core program depends only on these).
COLS, ROWS = 8, 4
M_CORE = 64          # output rows per core
N = 128              # output channels per pass
K_STEP = 64          # channels per window chunk / K step
MAX_WIDTH = 36       # widest supported image (window rows = 64 + 2*(W+2) + 2)
WIN_ROWS = M_CORE + 2 * (MAX_WIDTH + 2) + 2
GROUP_OUT_ROWS = M_CORE * ROWS * COLS
GROUP_ROWS = GROUP_OUT_ROWS + WIN_ROWS - M_CORE
RTP_LEN = 16


def stream_key(groups, wp, chunks):
    return f"g{groups}_w{wp}_c{chunks}"


def default_aie_kernels():
    import aie
    inc = Path(aie.__file__).resolve().parents[2] / "include" / "aie_kernels"
    return str(inc)


def geometry(batch, height, width):
    wp = width + 2
    m_total = batch * (height + 2) * wp
    groups = -(-m_total // GROUP_OUT_ROWS)
    return wp, m_total, groups


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def conv3x3w(
    X: In,
    Bw: In,
    C: Out,
    *,
    groups: CompileTime[int] = 29,
    wp: CompileTime[int] = 38,
    chunks: CompileTime[int] = 4,
    aie_kernels: CompileTime[str] = "",
    debug_static: CompileTime[int] = 0,
    b_l1_depth: CompileTime[int] = 2,
    slots: CompileTime[int] = 2,
):
    assert 3 <= wp <= MAX_WIDTH + 2 and groups >= 1 and chunks >= 1
    r, s, t = 8, 8, 8
    k = K_STEP
    K = 9 * k * chunks

    A_l2_ty = np.ndarray[(ROWS * WIN_ROWS, k), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(WIN_ROWS, k), np.dtype[bfloat16]]
    B_ty_l = np.ndarray[(k, N // 8), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(ROWS * M_CORE, N), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(M_CORE, N), np.dtype[bfloat16]]
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    flags = [f"-DDIM_M={M_CORE}", f"-DDIM_K={k}", f"-DDIM_N={N}", f"-I{aie_kernels}"]
    zero_kernel = ExternalFunction("zero_kernel_bf16", source_file=str(KERNEL_SRC),
                                   arg_types=[C_l1_ty], compile_flags=flags + ["-DZERO_ONLY"],
                                   use_chess=False)
    matmul_kernel = ExternalFunction("matmul_window", source_file=str(KERNEL_SRC),
                                     arg_types=[A_l1_ty, B_ty_l, C_l1_ty, np.int32],
                                     compile_flags=flags + ["-DWINDOW_ONLY", f"-DWIN_R={WIN_ROWS}"],
                                     use_chess=False)

    # Core receive: rows arrive row-major (WIN_ROWS x 64); store 8-channel
    # blocked [8][WIN_ROWS][8] so every 8x8 A block is contiguous.
    a_l1_dims = [(WIN_ROWS, 8), (k // 8, WIN_ROWS * 8), (8, 1)]
    c_l2l3_dims = [(M_CORE // r, r * N), (r, t), (N // t, r * t), (t, 1)]

    A_l3l2, A_l2l1, C_l2l3 = [], [], []
    C_l1l2 = [[None] * COLS for _ in range(ROWS)]
    for col in range(COLS):
        a = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=2)
        A_l3l2.append(a)
        A_l2l1.append(a.cons().split(
            tile=Tile(col, 1),
            offsets=[row * WIN_ROWS * k for row in range(ROWS)],
            obj_types=[A_l1_ty] * ROWS,
            names=[f"A_L2L1_{col}_{row}" for row in range(ROWS)],
            depths=[1] * ROWS,
            dims_from_stream=[a_l1_dims] * ROWS,
        ))
        c = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=2, dims_to_stream=c_l2l3_dims)
        C_l2l3.append(c)
        subs = c.prod().join([M_CORE * N * row for row in range(ROWS)], tile=Tile(col, 1),
                             obj_types=[C_l1_ty] * ROWS,
                             names=[f"C_L1L2_{col}_{row}" for row in range(ROWS)],
                             depths=[1] * ROWS)
        for row in range(ROWS):
            C_l1l2[row][col] = subs[row]
    B_l3l2 = ObjectFifo(B_ty_l, name="B_L3L2", depth=2)
    B_l2l1 = B_l3l2.cons().forward(tile=Tile(0, 1), obj_type=B_ty_l, name="B_L2L1",
                                   depth=b_l1_depth)

    rtps = [[Buffer(rtp_ty, name=f"rtp_{col}_{row}", initial_value=np.zeros(RTP_LEN, np.int32),
                    use_write_rtp=True) for col in range(COLS)] for row in range(ROWS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(COLS)] for _ in range(ROWS)]

    def core_fn(in_a, in_b, out_c, zero, matmul, rtp, barrier):
        barrier.wait_for_value(1)
        # debug_static: 1 = all static, 2 = static offsets, 3 = static loops
        if debug_static in (1, 3):
            n_groups, n_chunks = groups, chunks
        else:
            n_groups = rtp[0]
            n_chunks = rtp[1]
        if debug_static in (1, 2):
            offs = [dy * wp + dx for dy in range(3) for dx in range(3)]
        else:
            offs = [rtp[2 + i] for i in range(9)]
        # wait_for_value is an acquire-equal and leaves the lock at 1; move it
        # to 2 so the next dispatch blocks until its own set_barrier(1).
        # Releasing here, after the parameters are in registers, rather than
        # at the end of the dispatch avoids racing the next set_barrier.
        barrier.release_with_value(1)
        for _ in range_(n_groups):
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(n_chunks):
                win = in_a.acquire(1)
                for tap in range(9):
                    eb = in_b.acquire(1)
                    matmul(win, eb, elem_out, offs[tap])
                    in_b.release(1)
                in_a.release(1)
            out_c.release(1)

    workers = Worker.grid(ROWS, COLS, lambda row, col: Worker(
        core_fn,
        [A_l2l1[col][row].cons(), B_l2l1.cons(), C_l1l2[row][col].prod(),
         zero_kernel, matmul_kernel, rtps[row][col], barriers[row][col]],
        tile=Tile(col, 2 + row),
        stack_size=0xD00,
    ))

    x_elems = groups * chunks * GROUP_ROWS * k
    X_ty = np.ndarray[(x_elems,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(groups * GROUP_OUT_ROWS * N,), np.dtype[bfloat16]]
    B_tap = TensorTiler2D.group_tiler((1, K * N // 8), (1, K * N // 8), (1, 1))[0]
    C_taps = TensorTiler2D.group_tiler((groups * GROUP_OUT_ROWS, N), (ROWS * M_CORE, N), (1, 1))

    def a_tap(grp, col):
        # One descriptor per (group, column) for all channel chunks: the
        # leading entry is the shim BD iteration counter.  Always pass 4
        # entries; with 3 the compiler moves the core dimension into the
        # iteration slot (wrong data).  One task per group keeps the shim
        # channel's task queue (depth 4) from overflowing.
        base = grp * chunks * GROUP_ROWS * k + col * ROWS * M_CORE * k
        return TensorAccessPattern((x_elems,), base, [chunks, ROWS, WIN_ROWS, k],
                                   [GROUP_ROWS * k, M_CORE * k, k, 1])

    params = [groups, chunks] + [dy * wp + dx for dy in range(3) for dx in range(3)]

    def write_params(*bufs):
        for buf in bufs:
            for i, v in enumerate(params):
                buf[i] = v

    # Groups in flight.  Each group queues one task per shim channel, and a
    # shim channel's task queue holds 4: more silently drops tasks (hang).
    assert 1 <= slots <= 4
    slots_n = slots
    rt = Runtime()
    with rt.sequence(X_ty, B_ty, C_ty) as (x, b, c):
        rt.start(*[w for row in workers for w in row])
        rt.inline_ops(write_params, [buf for row in rtps for buf in row])
        for row in barriers:
            for bar in row:
                rt.set_barrier(bar, 1)
        tgs = [None] * slots_n
        open_slots = set()

        def finish(si):
            if si in open_slots:
                rt.finish_task_group(tgs[si])
                open_slots.discard(si)

        for grp in range(groups):
            si = grp % slots_n
            finish(si)
            tg = rt.task_group()
            tgs[si] = tg
            open_slots.add(si)
            rt.fill(B_l3l2.prod(), b, tap=B_tap, task_group=tg, wait=False, tile=Tile(0, 0))
            for col in range(COLS):
                rt.fill(A_l3l2[col].prod(), x, tap=a_tap(grp, col), task_group=tg,
                        wait=False, tile=Tile(col, 0))
            for col in range(COLS):
                rt.drain(C_l2l3[col].cons(), c, tap=C_taps[grp * COLS + col], task_group=tg,
                         wait=True, tile=Tile(col, 0))
        for i in range(1, slots_n + 1):
            finish((groups + i - 1) % slots_n)

    return Program(iron.get_current_device(), rt).resolve_program()


def build(out_dir, groups, wp, chunks, aie_kernels=None, debug_static=0, b_l1_depth=2,
          slots=2, keep_prj=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = stream_key(groups, wp, chunks)
    aie_kernels = aie_kernels or default_aie_kernels()
    tmp_x = out / f".{key}.xclbin.tmp"
    tmp_i = out / f".{key}.bin.tmp"
    from aie.iron.device import from_name
    from aie.utils.hostruntime import set_current_device
    set_current_device(from_name("npu2", n_cols=None))
    conv3x3w.specialize(groups=groups, wp=wp, chunks=chunks,
                        aie_kernels=aie_kernels, debug_static=debug_static,
                        b_l1_depth=b_l1_depth, slots=slots).compile(xclbin_path=str(tmp_x),
                                                         inst_path=str(tmp_i))
    shared = out / "conv3x3w.xclbin"
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
    ap.add_argument("--groups", type=int)
    ap.add_argument("--wp", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--aie-kernels", default=None)
    ap.add_argument("--keep-prj", action="store_true")
    ap.add_argument("--debug-static", type=int, default=0)
    ap.add_argument("--b-l1-depth", type=int, default=2)
    ap.add_argument("--slots", type=int, default=2)
    a = ap.parse_args()
    if a.groups is None:
        a.wp, _, a.groups = geometry(a.batch, a.height, a.width)
    print(build(a.out, a.groups, a.wp, a.chunks, a.aie_kernels, a.debug_static,
                a.b_l1_depth, a.slots, a.keep_prj))


if __name__ == "__main__":
    main()
