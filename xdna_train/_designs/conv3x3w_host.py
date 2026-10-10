"""Reference (numpy) host layouts for the conv3x3w NPU family.

These are the executable specification for the C++ packing in
xdna_train/_native; tests compare the two.  Pure numpy, no torch, so the
hardware tests can run under the system Python that provides pyxrt.
"""
import numpy as np

COLS, ROWS, M_CORE, N, K_STEP, MAX_WIDTH = 8, 4, 64, 128, 64, 36
WIN_ROWS = M_CORE + 2 * (MAX_WIDTH + 2) + 2
GROUP_OUT_ROWS = M_CORE * ROWS * COLS
GROUP_ROWS = GROUP_OUT_ROWS + WIN_ROWS - M_CORE


def geometry(batch, height, width):
    wp = width + 2
    m_total = batch * (height + 2) * wp
    groups = -(-m_total // GROUP_OUT_ROWS)
    return wp, m_total, groups


def bf16_bits(x):
    """float32 -> bfloat16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)


def bf16_value(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def pack_input(x_nchw, chunks):
    """NCHW float -> [groups][chunks][GROUP_ROWS][64] bf16 bits.

    Rows are padded-grid NHWC pixels (zero border), each group carries its own
    halo, channels are zero-padded to chunks*64.
    """
    B, C, H, W = x_nchw.shape
    wp, m_total, groups = geometry(B, H, W)
    rows = np.zeros((groups * GROUP_OUT_ROWS + WIN_ROWS, chunks * K_STEP), np.uint16)
    grid = np.zeros((B, H + 2, wp, chunks * K_STEP), np.uint16)
    grid[:, 1:H + 1, 1:W + 1, :C] = bf16_bits(x_nchw.transpose(0, 2, 3, 1))
    rows[:m_total] = grid.reshape(m_total, -1)
    idx = np.arange(groups)[:, None] * GROUP_OUT_ROWS + np.arange(GROUP_ROWS)[None, :]
    out = rows[idx].reshape(groups, GROUP_ROWS, chunks, K_STEP).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(out)


def atb_shuffle(b, k=K_STEP, n=N):
    """AMD gemm_atb layout_transpose_L1_1x2_8x8block for a (K, N) matrix."""
    K, Nn = b.shape
    t = b.reshape(K // k, k, Nn // n, n).transpose(2, 0, 1, 3)
    t = t.reshape(Nn // n, K // k, k // 8, 8, n // 16, 2, 8).transpose(0, 1, 4, 2, 5, 6, 3)
    return np.ascontiguousarray(t).reshape(-1)


def bfp16_encode(x):
    """bfp16ebs8 bytes, blocks of 8 consecutive values: shared exponent byte
    (block max) + 8 int8 mantissas, round to nearest (AMD's reference helper
    truncates, which biases sums by ~16% at K=2304)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).reshape(-1, 8)
    max_exp = ((u >> 23) & 0xFF).astype(np.int64).max(axis=1, keepdims=True)
    lsb = np.ldexp(1.0, max_exp - 127 - 6)
    q = np.clip(np.rint(x.reshape(-1, 8).astype(np.float64) / lsb), -128, 127).astype(np.int64)
    out = np.empty((u.shape[0], 9), np.uint8)
    out[:, 0] = max_exp[:, 0].astype(np.uint8)
    out[:, 1:] = (q & 0xFF).astype(np.uint8)
    return out.reshape(-1)


def pack_weights(w_oihw, chunks, n_block=0):
    """[Cout, Cin, 3, 3] -> bfp16 B for output channels [128*n_block, +128).

    K order is (chunk, dy, dx, c); input channels and output channels are
    zero-padded to the family shape.
    """
    Cout, Cin = w_oihw.shape[:2]
    wk = np.zeros((chunks * K_STEP, 3, 3, N), np.float32)
    lo, hi = N * n_block, min(Cout, N * (n_block + 1))
    wk[:Cin, :, :, :hi - lo] = w_oihw[lo:hi].transpose(1, 2, 3, 0)
    wk = wk.reshape(chunks, K_STEP, 3, 3, N).transpose(0, 2, 3, 1, 4).reshape(-1, N)
    return bfp16_encode(atb_shuffle(wk))


def unpack_output(c_bits, batch, height, width, cout):
    """[groups*2048][128] bf16 bits -> NCHW float32 (valid pixels/channels)."""
    wp, m_total, _ = geometry(batch, height, width)
    c = bf16_value(c_bits.reshape(-1, N)[:m_total])
    c = c.reshape(batch, height + 2, wp, N)[:, :height, :width, :cout]
    return np.ascontiguousarray(c.transpose(0, 3, 1, 2))
