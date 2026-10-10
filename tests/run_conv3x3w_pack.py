#!/usr/bin/python3
"""Compare the C++ conv3x3w host layouts against conv3x3w_host.py (bit-exact)."""
import ctypes, os, subprocess, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "xdna_train", "_designs"))
import conv3x3w_host as ref

tmp = tempfile.mkdtemp()
so = os.path.join(tmp, "conv_pack.so")
subprocess.check_call(["g++", "-O2", "-std=c++20", "-fopenmp", "-shared", "-fPIC",
                       os.path.join(ROOT, "xdna_train", "_native", "conv_pack.cpp"), "-o", so])
lib = ctypes.CDLL(so)
lib.xdna_conv_set_threads(8)
P = ctypes.c_void_p
I = ctypes.c_int
lib.xdna_conv3x3w_pack_input_bf16.argtypes = [P, P, I, I, I, I, I]
lib.xdna_conv3x3w_pack_weights_bfp16.argtypes = [P, P, I, I, I, I]
lib.xdna_conv3x3w_pack_weights_dx_bfp16.argtypes = [P, P, I, I, I, I]
lib.xdna_conv3x3w_unpack_output_bf16.argtypes = [P, P, I, I, I, I, I, I]

rng = np.random.default_rng(0)
fails = 0


def timed(fn, reps=20):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1e3


def report(name, ok, ms):
    global fails
    fails += not ok
    print(f"  {name:8s} {'OK  ' if ok else 'FAIL'} {ms:8.3f} ms")


def case(B, C, H, W, Cout):
    global fails
    print(f"case B={B} C={C} H={H} W={W} Cout={Cout}")
    chunks = -(-C // 64)
    wp, m_total, groups = ref.geometry(B, H, W)
    # input: buffer zero-filled once, reused with two different inputs
    shape = (groups, chunks, ref.GROUP_ROWS, 64)
    dst = np.zeros(shape, np.uint16)
    ok = True
    for _ in range(2):
        x = rng.standard_normal((B, C, H, W)).astype(np.float32)
        xb = np.ascontiguousarray(ref.bf16_bits(x))
        lib.xdna_conv3x3w_pack_input_bf16(xb.ctypes.data, dst.ctypes.data, B, C, H, W, chunks)
        ok &= np.array_equal(dst, ref.pack_input(x, chunks))
    ms = timed(lambda: lib.xdna_conv3x3w_pack_input_bf16(xb.ctypes.data, dst.ctypes.data, B, C, H, W, chunks))
    report("input", bool(ok), ms)

    # weights
    w = (rng.standard_normal((Cout, C, 3, 3)) * rng.choice([1e-3, 1, 50], (Cout, C, 1, 1))).astype(np.float32)
    w[:, :, 0, 0] = np.where(rng.random((Cout, C)) < 0.3, 0, w[:, :, 0, 0])
    w[:, :, 1, 1] = rng.integers(-40, 40, (Cout, C)) / 8.0          # exact ties at 1/8 steps
    w[:, : min(C, 8), :, :] = 0                                        # all-zero blocks
    wbits = ref.bf16_bits(w)
    wf = ref.bf16_value(wbits)
    wb = np.ascontiguousarray(wbits)
    ok = True
    ms = 0
    for nb in range(-(-Cout // 128)):
        expect = ref.pack_weights(wf, chunks, nb)
        out = np.zeros(expect.shape, np.uint8)
        f = lambda: lib.xdna_conv3x3w_pack_weights_bfp16(wb.ctypes.data, out.ctypes.data, Cout, C, chunks, nb)
        ms = max(ms, timed(f, 5))
        ok &= np.array_equal(out, expect)
    report("weights", bool(ok), ms)

    # dX weights: reference applied to flip(w, {2,3}).transpose(0,1)
    wdx = np.ascontiguousarray(wf[:, :, ::-1, ::-1].transpose(1, 0, 2, 3))
    ok = True
    ms = 0
    for nb in range(-(-C // 128)):
        expect = ref.pack_weights(wdx, -(-Cout // 64), nb)
        out = np.zeros(expect.shape, np.uint8)
        f = lambda: lib.xdna_conv3x3w_pack_weights_dx_bfp16(wb.ctypes.data, out.ctypes.data, Cout, C, -(-Cout // 64), nb)
        ms = max(ms, timed(f, 5))
        ok &= np.array_equal(out, expect)
    report("weightdx", bool(ok), ms)

    # unpack
    c = rng.integers(0, 65536, (groups * ref.GROUP_OUT_ROWS, 128), dtype=np.uint16)
    expect = ref.unpack_output(c, B, H, W, Cout)
    ctot, cs = Cout + 7, 3
    full = np.zeros((B, ctot, H, W), np.float32)
    outb = np.zeros((B, ctot, H, W), np.uint16)
    nv = min(Cout, 128)
    f = lambda: lib.xdna_conv3x3w_unpack_output_bf16(c.ctypes.data, outb.ctypes.data, B, H, W, nv, ctot, cs)
    ms = timed(f)
    full = ref.bf16_value(outb)
    ok = np.array_equal(full[:, cs:cs + nv].view(np.uint32), expect[:, :nv].view(np.uint32))
    ok &= not outb[:, :cs].any() and not outb[:, cs + nv:].any()
    report("unpack", bool(ok), ms)


for s in [(40, 256, 36, 36, 128), (40, 64, 36, 36, 64), (4, 195, 36, 36, 64),
          (40, 128, 9, 9, 128), (40, 256, 18, 18, 256), (3, 3, 5, 7, 10)]:
    case(*s)
print("FAILED" if fails else "ALL OK")
sys.exit(1 if fails else 0)
