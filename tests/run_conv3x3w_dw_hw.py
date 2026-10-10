#!/usr/bin/env python3
"""Hardware check of conv3x3w_dw (weight gradient) against an FP32 reference.

System python with pyxrt.  usage: run_conv3x3w_dw_hw.py ART_DIR [B,H,W,Cin,Cout ...]
ART_DIR holds conv3x3w_dw.xclbin and the per-pass streams
(xdna_train/_designs/conv3x3w_dw.py stream_key).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyxrt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xdna_train" / "_designs"))
import conv3x3w_host as h  # noqa: E402

ACC = 3 * 64 * 32


def dw_ref(x, dy):
    B, C, H, W = x.shape
    xp = np.zeros((B, C, H + 2, W + 2), np.float32)
    xp[:, :, 1:-1, 1:-1] = x
    out = np.zeros((dy.shape[1], C, 3, 3), np.float32)
    for ky in range(3):
        for kx in range(3):
            out[:, :, ky, kx] = np.einsum("bohw,bihw->oi", dy, xp[:, :, ky:ky + H, kx:kx + W],
                                          optimize=True)
    return out


def unpack_pass(c, dw, ci_pass, co_pass, Cin, Cout):
    """C [6 col][4 row][3 dx][8 ci blk][4 co blk][8 ci][8 co] -> dW slice."""
    t = c.reshape(2, 3, 4, 3, 8, 4, 8, 8)  # chunk_l, dy, row, dx, cib, cob, ci, co
    for cl in range(2):
        for row in range(4):
            ci0 = (2 * ci_pass + cl) * 64
            co0 = (2 * co_pass + row // 2) * 64 + (row % 2) * 32
            if ci0 >= Cin or co0 >= Cout:
                continue
            blk = t[cl, :, row]  # dy, dx, cib, cob, ci, co
            blk = blk.transpose(0, 1, 2, 4, 3, 5).reshape(3, 3, 64, 32)  # dy dx ci co
            nci, nco = min(64, Cin - ci0), min(32, Cout - co0)
            dw[co0:co0 + nco, ci0:ci0 + nci] = blk[:, :, :nci, :nco].transpose(3, 2, 0, 1)


def main():
    art = Path(sys.argv[1])
    shapes = [tuple(int(v) for v in s.split(",")) for s in sys.argv[2:]] or [(40, 36, 36, 256, 128)]
    dev = pyxrt.device(0)
    xclbin = pyxrt.xclbin(str(art / "conv3x3w_dw.xclbin"))
    dev.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(dev, xclbin.get_uuid())
    kern = pyxrt.kernel(ctx, "MLIR_AIE")
    rng = np.random.default_rng(0)
    ok = True
    for (B, H, W, Cin, Cout) in shapes:
        wp, _, groups = h.geometry(B, H, W)
        chunks, cochunks = -(-Cin // 64), -(-Cout // 64)
        x = rng.standard_normal((B, Cin, H, W), dtype=np.float32)
        dy = rng.standard_normal((B, Cout, H, W), dtype=np.float32)
        xb, db = h.bf16_value(h.bf16_bits(x)), h.bf16_value(h.bf16_bits(dy))
        xs, ds = h.pack_input(xb, chunks), h.pack_input(db, cochunks)
        bo_x = pyxrt.bo(dev, xs.nbytes, pyxrt.bo.host_only, kern.group_id(3))
        bo_d = pyxrt.bo(dev, ds.nbytes, pyxrt.bo.host_only, kern.group_id(4))
        bo_c = pyxrt.bo(dev, 6 * 4 * ACC * 4, pyxrt.bo.host_only, kern.group_id(5))
        bo_x.write(xs.tobytes(), 0)
        bo_d.write(ds.tobytes(), 0)
        for bo in (bo_x, bo_d):
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        dw = np.zeros((Cout, Cin, 3, 3), np.float32)
        total = 0.0
        state = None
        for p in range(-(-chunks // 2)):
            for q in range(-(-cochunks // 2)):
                key = f"g{groups}_w{wp}_c{chunks}_d{cochunks}_p{p}_q{q}"
                instr = np.fromfile(art / f"{key}.bin", dtype=np.uint32)
                bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
                bo_i.write(instr.tobytes(), 0)
                bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                times = []
                for _ in range(int(__import__("os").environ.get("DW_REPS", "4"))):
                    t0 = time.perf_counter()
                    state = kern(3, bo_i, len(instr), bo_x, bo_d, bo_c).wait()
                    times.append(time.perf_counter() - t0)
                total += np.median(times[1:] if len(times) > 1 else times)
                bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                c = np.frombuffer(bo_c.read(6 * 4 * ACC * 4, 0), dtype=np.float32)
                unpack_pass(c, dw, p, q, Cin, Cout)
        ref = dw_ref(xb, db)
        err = np.sqrt(((dw - ref) ** 2).mean() / (ref ** 2).mean())
        good = state == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED and err < 1e-3
        ok &= good
        gf = 2 * B * H * W * 9 * Cin * Cout / 1e9
        print(f"dW B{B} {H}x{W} Cin{Cin} Cout{Cout}: {1e3 * total:.2f} ms "
              f"({gf / total / 1e3:.2f} TF/s useful) rel RMS {err:.2e} {'OK' if good else 'FAIL'}",
              flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
