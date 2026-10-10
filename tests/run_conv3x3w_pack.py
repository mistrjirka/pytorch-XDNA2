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
lib.xdna_conv3x3w_pack_input_range_bf16.argtypes = [P, P, I, I, I, I, I, I, I]
lib.xdna_conv3x3w_unpack_output_range_bf16.argtypes = [P, P, I, I, I, I, I, I, I, I]

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

def splits(groups):
    out = {"1": [groups], "each": [1] * groups}
    out["2eq"] = [groups - groups // 2, groups // 2] if groups > 1 else [groups]
    if groups >= 3:
        a = max(1, groups // 5)
        b = max(1, groups // 3)
        out["3uneven"] = [a, b, groups - a - b] if groups - a - b > 0 else [groups]
    return {k: [x for x in v if x > 0] for k, v in out.items()}


def range_case(B, C, H, W, Cout):
    print(f"range case B={B} C={C} H={H} W={W} Cout={Cout}")
    chunks = -(-C // 64)
    wp, m_total, groups = ref.geometry(B, H, W)
    x = rng.standard_normal((B, C, H, W)).astype(np.float32)
    xb = np.ascontiguousarray(ref.bf16_bits(x))
    full_ref = ref.pack_input(x, chunks)
    c = rng.integers(0, 65536, (groups * ref.GROUP_OUT_ROWS, 128), dtype=np.uint16)
    expect = ref.unpack_output(c, B, H, W, Cout)
    nv, ctot, cs = min(Cout, 128), Cout + 7, 3
    ok_in = ok_out = True
    for name, parts in splits(groups).items():
        g = 0
        outb = np.zeros((B, ctot, H, W), np.uint16)
        for n in parts:
            dst = np.zeros((n, chunks, ref.GROUP_ROWS, 64), np.uint16)
            for _ in range(2):  # reuse without re-zeroing
                lib.xdna_conv3x3w_pack_input_range_bf16(xb.ctypes.data, dst.ctypes.data, B, C, H, W, chunks, g, n)
            ok_in &= np.array_equal(dst, full_ref[g:g + n])
            p0, p1 = g * ref.GROUP_OUT_ROWS, (g + n) * ref.GROUP_OUT_ROWS
            cs_ = np.ascontiguousarray(c[p0:p1])
            lib.xdna_conv3x3w_unpack_output_range_bf16(cs_.ctypes.data, outb.ctypes.data, B, H, W, nv, ctot, cs, p0, p1)
            g += n
        got = ref.bf16_value(outb)
        ok_out &= np.array_equal(got[:, cs:cs + nv].view(np.uint32), expect[:, :nv].view(np.uint32))
        ok_out &= not outb[:, :cs].any() and not outb[:, cs + nv:].any()
    report("rng-in", bool(ok_in), 0)
    report("rng-out", bool(ok_out), 0)
    # mid-row ranges: arbitrary p splits, tiny piece sizes
    cuts = sorted(set([0, 1, wp + 3, 5 * wp - 1, m_total // 2, m_total - 1, groups * ref.GROUP_OUT_ROWS]))
    outb = np.zeros((B, ctot, H, W), np.uint16)
    for a, b_ in zip(cuts[:-1], cuts[1:]):
        cs_ = np.ascontiguousarray(c[a:b_])
        lib.xdna_conv3x3w_unpack_output_range_bf16(cs_.ctypes.data, outb.ctypes.data, B, H, W, nv, ctot, cs, a, b_)
    got = ref.bf16_value(outb)
    report("rng-mid", bool(np.array_equal(got[:, cs:cs + nv].view(np.uint32), expect[:, :nv].view(np.uint32))), 0)


def range_timing():
    B, C, H, W = 40, 256, 36, 36
    chunks = 4
    wp, m_total, groups = ref.geometry(B, H, W)
    xb = np.ascontiguousarray(ref.bf16_bits(rng.standard_normal((B, C, H, W)).astype(np.float32)))
    c = rng.integers(0, 65536, (groups * ref.GROUP_OUT_ROWS, 128), dtype=np.uint16)
    dst = np.zeros((groups, chunks, ref.GROUP_ROWS, 64), np.uint16)
    outb = np.zeros((B, 128, H, W), np.uint16)
    print(f"timing B=40 C=256 36x36 groups={groups} (8 threads)")
    print(f"  pack full   {timed(lambda: lib.xdna_conv3x3w_pack_input_bf16(xb.ctypes.data, dst.ctypes.data, B, C, H, W, chunks)):.3f} ms")
    print(f"  unpack full {timed(lambda: lib.xdna_conv3x3w_unpack_output_bf16(c.ctypes.data, outb.ctypes.data, B, H, W, 128, 128, 0)):.3f} ms")
    for k in (2, 4):
        per = -(-groups // k)
        parts = [(g, min(per, groups - g)) for g in range(0, groups, per)]
        def fp():
            for g, n in parts:
                lib.xdna_conv3x3w_pack_input_range_bf16(xb.ctypes.data, dst[g:g + n].ctypes.data, B, C, H, W, chunks, g, n)
        def fu():
            for g, n in parts:
                p0 = g * ref.GROUP_OUT_ROWS
                lib.xdna_conv3x3w_unpack_output_range_bf16(c[p0:].ctypes.data, outb.ctypes.data, B, H, W, 128, 128, 0, p0, p0 + n * ref.GROUP_OUT_ROWS)
        print(f"  {k} pieces: pack total {timed(fp):.3f} ms, unpack total {timed(fu):.3f} ms")


for s in [(40, 256, 36, 36, 128), (3, 70, 5, 7, 10), (4, 195, 36, 36, 64), (40, 128, 9, 9, 128)]:
    range_case(*s)
range_timing()
print("FAILED" if fails else "ALL OK")
sys.exit(1 if fails else 0)
