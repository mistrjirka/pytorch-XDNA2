#!/usr/bin/env python3
"""Hardware check of the conv3x3w family: several shapes on ONE xclbin.

Runs with the system Python (pyxrt).  For each shape: pack input/weights with
the numpy reference layouts, dispatch the shape's stream on the shared xclbin,
compare to an FP32 conv of the same BF16 inputs.

usage: run_conv3x3w_hw.py ARTIFACT_DIR [shape ...]   shape = B,H,W,Cin,Cout
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyxrt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xdna_train" / "_designs"))
import conv3x3w_host as h  # noqa: E402


def conv_ref(x, w):
    B, C, H, W = x.shape
    xp = np.zeros((B, C, H + 2, W + 2), np.float32)
    xp[:, :, 1:-1, 1:-1] = x
    out = np.zeros((B, w.shape[0], H, W), np.float32)
    for dy in range(3):
        for dx in range(3):
            patch = xp[:, :, dy:dy + H, dx:dx + W]
            out += np.einsum("bchw,oc->bohw", patch, w[:, :, dy, dx], optimize=True)
    return out


def main():
    art = Path(sys.argv[1])
    shapes = [tuple(int(v) for v in s.split(",")) for s in sys.argv[2:]] or [(40, 36, 36, 256, 128)]
    dev = pyxrt.device(0)
    xclbin = pyxrt.xclbin(str(art / "conv3x3w.xclbin"))
    dev.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(dev, xclbin.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    rng = np.random.default_rng(0)
    ok = True
    for (B, H, W, Cin, Cout) in shapes:
        chunks = -(-Cin // h.K_STEP)
        key = f"b{B}_h{H}_w{W}_c{chunks}"
        instr = np.fromfile(art / f"{key}.bin", dtype=np.uint32)
        x = rng.standard_normal((B, Cin, H, W), dtype=np.float32)
        x = np.where(x > 0, x, 0.2 * x)
        w = rng.standard_normal((Cout, Cin, 3, 3), dtype=np.float32) * np.sqrt(2.0 / (9 * Cin))
        xb, wb = h.bf16_value(h.bf16_bits(x)), h.bf16_value(h.bf16_bits(w))
        xs = h.pack_input(xb, chunks)
        _, _, groups = h.geometry(B, H, W)
        c_elems = groups * h.GROUP_OUT_ROWS * h.N
        bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
        bo_x = pyxrt.bo(dev, xs.nbytes, pyxrt.bo.host_only, kern.group_id(3))
        bo_c = pyxrt.bo(dev, c_elems * 2, pyxrt.bo.host_only, kern.group_id(5))
        bo_i.write(instr.tobytes(), 0)
        bo_x.write(xs.tobytes(), 0)
        for bo in (bo_i, bo_x):
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        outs, times = [], []
        for nb in range(-(-Cout // h.N)):
            bw = h.pack_weights(wb, chunks, nb)
            bo_b = pyxrt.bo(dev, bw.nbytes, pyxrt.bo.host_only, kern.group_id(4))
            bo_b.write(bw.tobytes(), 0)
            bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            for i in range(4):
                t0 = time.perf_counter()
                run = kern(3, bo_i, len(instr), bo_x, bo_b, bo_c)
                state = run.wait()
                times.append(time.perf_counter() - t0)
            bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            cbits = np.frombuffer(bo_c.read(c_elems * 2, 0), dtype=np.uint16)
            outs.append(h.unpack_output(cbits, B, H, W, min(h.N, Cout - nb * h.N)))
        y = np.concatenate(outs, axis=1)
        ref = conv_ref(xb, wb)
        err = np.sqrt(((y - ref) ** 2).mean() / (ref ** 2).mean())
        ms = 1e3 * np.median(times[1:])
        gf = 2 * B * H * W * 9 * Cin * Cout / 1e9 / -(-Cout // h.N)
        good = state == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED and err < 0.02
        ok &= good
        print(f"{key} Cin{Cin} Cout{Cout}: {state.name} {ms:.2f} ms/pass ({gf / ms:.2f} TF/s) "
              f"rel RMS {100 * err:.2f}% {'OK' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
