#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// Width of the pack loops only.  Set through num_threads() on each loop so the
// caller's OpenMP width (PyTorch's intra-op setting) is never modified.
static int g_pack_threads = 1;

extern "C" {

// NCHW bf16 -> [H+2,W+2,C/32,B,32] halo layout used by virtual-im2col fwd/dX.
void xdna_pack_conv3x3_halo_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int H,
    int W) {
  const int cb = C / 32;
  const std::size_t elems =
      std::size_t(H + 2) * (W + 2) * cb * B * 32;
  std::memset(dst, 0, elems * sizeof(uint16_t));

#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int cblock = 0; cblock < cb; ++cblock) {
        for (int b = 0; b < B; ++b) {
          const uint16_t* s =
              src + ((std::size_t(b) * C + cblock * 32) * H + y) * W + x;
          uint16_t* d =
              dst + ((((std::size_t(y + 1) * (W + 2) + (x + 1)) * cb + cblock) * B + b) * 32);
          for (int lane = 0; lane < 32; ++lane)
            d[lane] = s[std::size_t(lane) * H * W];
        }
      }
    }
  }
}

// NCHW bf16 -> [H+2,W_sched+2,C/32,B,32] with right-side spatial
// padding. This lets virtual-im2col use an X tile width that does not divide
// the logical image width while PyTorch later views only the real W columns.
void xdna_pack_conv3x3_halo_padded_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int H,
    int W,
    int W_sched) {
  const int cb = C / 32;
  const std::size_t elems =
      std::size_t(H + 2) * (W_sched + 2) * cb * B * 32;
  std::memset(dst, 0, elems * sizeof(uint16_t));

#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int cblock = 0; cblock < cb; ++cblock) {
        for (int b = 0; b < B; ++b) {
          const uint16_t* srcp =
              src + ((std::size_t(b) * C + cblock * 32) * H + y) * W + x;
          uint16_t* dstp =
              dst + ((((std::size_t(y + 1) * (W_sched + 2) + (x + 1)) * cb
                        + cblock) * B + b) * 32);
          for (int lane = 0; lane < 32; ++lane)
            dstp[lane] = srcp[std::size_t(lane) * H * W];
        }
      }
    }
  }
}

// OIHW bf16 -> [ky,kx,Cin,Cout] == GEMM [9*Cin,Cout].

// Compact channels-last logical NCHW ([B,H,W,C] physical) -> the padded
// [H+2,W_sched+2,C/32,B,32] virtual-im2col source. Channel blocks are
// contiguous in channels-last storage, so each 32-channel lane group is one
// memcpy instead of a strided gather.
void xdna_pack_conv3x3_halo_padded_channels_last_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int H,
    int W,
    int W_sched) {
  const int cb = C / 32;
  const std::size_t elems =
      std::size_t(H + 2) * (W_sched + 2) * cb * B * 32;
  std::memset(dst, 0, elems * sizeof(uint16_t));

#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int cblock = 0; cblock < cb; ++cblock) {
        for (int b = 0; b < B; ++b) {
          const uint16_t* srcp =
              src + (((std::size_t(b) * H + y) * W + x) * C + cblock * 32);
          uint16_t* dstp =
              dst + ((((std::size_t(y + 1) * (W_sched + 2) + (x + 1)) * cb
                        + cblock) * B + b) * 32);
          std::memcpy(dstp, srcp, 32 * sizeof(uint16_t));
        }
      }
    }
  }
}


// Two compact channels-last tensors are logically concatenated along C and
// nearest-upsampled by 2, then packed directly into the Conv virtual-im2col
// halo layout. This deliberately avoids materializing either the concatenated
// or the upsampled PyTorch tensor. C0/C1 must be multiples of 32.
void xdna_pack_conv3x3_halo_cat2_upsample2_channels_last_bf16(
    const uint16_t* src0,
    int C0,
    const uint16_t* src1,
    int C1,
    uint16_t* dst,
    int B,
    int H,
    int W,
    int W_sched) {
  const int C = C0 + C1;
  const int cb0 = C0 / 32;
  const int cb = C / 32;
  const int OH = H * 2;
  const int OW = W * 2;
  const std::size_t elems =
      std::size_t(OH + 2) * (W_sched + 2) * cb * B * 32;
  std::memset(dst, 0, elems * sizeof(uint16_t));

#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int oy = 0; oy < OH; ++oy) {
    for (int ox = 0; ox < OW; ++ox) {
      const int sy = oy >> 1;
      const int sx = ox >> 1;
      for (int cblock = 0; cblock < cb; ++cblock) {
        const bool first = cblock < cb0;
        const int local_block = first ? cblock : cblock - cb0;
        const int SC = first ? C0 : C1;
        const uint16_t* src = first ? src0 : src1;
        for (int b = 0; b < B; ++b) {
          const uint16_t* srcp =
              src + (((std::size_t(b) * H + sy) * W + sx) * SC +
                     local_block * 32);
          uint16_t* dstp =
              dst + ((((std::size_t(oy + 1) * (W_sched + 2) + (ox + 1)) *
                        cb + cblock) * B + b) * 32);
          std::memcpy(dstp, srcp, 32 * sizeof(uint16_t));
        }
      }
    }
  }
}

// NPU GEMM output is [Y,X_sched,B,C]. Reorder only the real W columns into
// compact channels-last PyTorch storage [B,H,W,C]. Both source and destination
// have C contiguous, so this is a row of small contiguous C-channel copies,
// not a scalar transpose.
void xdna_yxbc_to_channels_last_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int H,
    int W,
    int W_sched) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const uint16_t* srcp =
            src + (((std::size_t(y) * W_sched + x) * B + b) * C);
        uint16_t* dstp =
            dst + (((std::size_t(b) * H + y) * W + x) * C);
        std::memcpy(dstp, srcp, std::size_t(C) * sizeof(uint16_t));
      }
    }
  }
}

// [Y*X,B,C] NPU result -> channels [c_start, c_start+C) of NCHW dst with
// C_total channels.  Blocked 32x64 tiles keep the strided gather in cache;
// ATen's generic permuted copy_ is several times slower for this transpose.
void xdna_yxbc_to_nchw_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int HW,
    int C_total,
    int c_start) {
  constexpr int kTc = 32;
  constexpr int kTp = 64;
  const int c_tiles = (C + kTc - 1) / kTc;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int ct = 0; ct < c_tiles; ++ct) {
      const int c0 = ct * kTc;
      const int c1 = c0 + kTc < C ? c0 + kTc : C;
      for (int p0 = 0; p0 < HW; p0 += kTp) {
        const int p1 = p0 + kTp < HW ? p0 + kTp : HW;
        for (int c = c0; c < c1; ++c) {
          uint16_t* d =
              dst + (std::size_t(b) * C_total + c_start + c) * HW;
          for (int p = p0; p < p1; ++p)
            d[p] = src[(std::size_t(p) * B + b) * C + c];
        }
      }
    }
  }
}

// Scatter one [Y,X_sched,B,Cslice] NPU result into a channel slice of compact
// channels-last logical [B,H,W,Ctotal] storage.
void xdna_yxbc_slice_to_channels_last_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Ctotal,
    int Cslice,
    int channel_offset,
    int H,
    int W,
    int W_sched) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const uint16_t* srcp =
            src + (((std::size_t(y) * W_sched + x) * B + b) * Cslice);
        uint16_t* dstp =
            dst + (((std::size_t(b) * H + y) * W + x) * Ctotal +
                   channel_offset);
        std::memcpy(dstp, srcp, std::size_t(Cslice) * sizeof(uint16_t));
      }
    }
  }
}

// Pack one spatial stripe from compact channels-last [B,H,W,C] into the
// virtual-im2col halo layout. The stripe halo references the *global* source
// rows, so splitting the output into stripes does not introduce artificial
// zero-padding at stripe boundaries.
void xdna_pack_conv3x3_halo_padded_channels_last_stripe_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int full_h,
    int W,
    int y0,
    int stripe_h,
    int W_sched) {
  const int cb = C / 32;
  const std::size_t elems =
      std::size_t(stripe_h + 2) * (W_sched + 2) * cb * B * 32;
  std::memset(dst, 0, elems * sizeof(uint16_t));

#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int ly = 0; ly < stripe_h + 2; ++ly) {
    for (int x = 0; x < W; ++x) {
      const int gy = y0 + ly - 1;
      if (gy < 0 || gy >= full_h)
        continue;
      for (int cblock = 0; cblock < cb; ++cblock) {
        for (int b = 0; b < B; ++b) {
          const uint16_t* srcp =
              src + (((std::size_t(b) * full_h + gy) * W + x) * C +
                     cblock * 32);
          uint16_t* dstp =
              dst + ((((std::size_t(ly) * (W_sched + 2) + (x + 1)) * cb +
                        cblock) * B + b) * 32);
          std::memcpy(dstp, srcp, 32 * sizeof(uint16_t));
        }
      }
    }
  }
}

// Scatter one stripe result [stripe_h,W_sched,B,Cslice] into a channel slice
// of full compact channels-last [B,full_h,W,Ctotal] storage.
void xdna_yxbc_slice_stripe_to_channels_last_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Ctotal,
    int Cslice,
    int channel_offset,
    int full_h,
    int y0,
    int stripe_h,
    int W,
    int W_sched) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int ly = 0; ly < stripe_h; ++ly) {
      const int gy = y0 + ly;
      for (int x = 0; x < W; ++x) {
        const uint16_t* srcp =
            src + (((std::size_t(ly) * W_sched + x) * B + b) * Cslice);
        uint16_t* dstp =
            dst + (((std::size_t(b) * full_h + gy) * W + x) * Ctotal +
                   channel_offset);
        std::memcpy(dstp, srcp, std::size_t(Cslice) * sizeof(uint16_t));
      }
    }
  }
}

void xdna_pack_conv3x3_weight_fwd_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int Cout,
    int Cin) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      for (int ci = 0; ci < Cin; ++ci) {
        uint16_t* d =
            dst + (std::size_t((ky * 3 + kx) * Cin + ci) * Cout);
        for (int co = 0; co < Cout; ++co) {
          d[co] = src[((std::size_t(co) * Cin + ci) * 3 + ky) * 3 + kx];
        }
      }
    }
  }
}

// OIHW bf16 -> dX conv weights [ky,kx,Cout,Cin], with spatial flip and IO swap.
void xdna_pack_conv3x3_weight_dx_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int Cout,
    int Cin) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      for (int co = 0; co < Cout; ++co) {
        uint16_t* d =
            dst + (std::size_t((ky * 3 + kx) * Cout + co) * Cin);
        for (int ci = 0; ci < Cin; ++ci) {
          d[ci] = src[((std::size_t(co) * Cin + ci) * 3 + (2 - ky)) * 3 + (2 - kx)];
        }
      }
    }
  }
}

// NCHW saved activation -> padded row-major dW A matrix.
// Real rows are [ky,kx,Cin], real columns are [y,x,b].
// Destination is zero-padded to Mpad x Kpad.
void xdna_pack_conv3x3_dw_a_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Cin,
    int H,
    int W,
    int Mpad,
    int Kpad) {
  std::memset(dst, 0, std::size_t(Mpad) * Kpad * sizeof(uint16_t));
  const int real_rows = 9 * Cin;

#pragma omp parallel for num_threads(g_pack_threads) schedule(static)
  for (int row = 0; row < real_rows; ++row) {
    const int q = row / Cin;
    const int ci = row - q * Cin;
    const int ky = q / 3;
    const int kx = q - ky * 3;
    uint16_t* d = dst + std::size_t(row) * Kpad;

    int p = 0;
    for (int y = 0; y < H; ++y) {
      const int iy = y + ky - 1;
      for (int x = 0; x < W; ++x) {
        const int ix = x + kx - 1;
        if (unsigned(iy) < unsigned(H) && unsigned(ix) < unsigned(W)) {
          for (int b = 0; b < B; ++b) {
            d[p + b] =
                src[((std::size_t(b) * Cin + ci) * H + iy) * W + ix];
          }
        }
        p += B;
      }
    }
  }
}

// Compact channels-last dY [B,H,W,Cout] -> GEMM B [y,x,b,Cout].
void xdna_pack_conv3x3_dy_channels_last_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Cout,
    int H,
    int W) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int b = 0; b < B; ++b) {
        const uint16_t* srcp =
            src + (((std::size_t(b) * H + y) * W + x) * Cout);
        uint16_t* dstp =
            dst + (((std::size_t(y) * W + x) * B + b) * Cout);
        std::memcpy(dstp, srcp, std::size_t(Cout) * sizeof(uint16_t));
      }
    }
  }
}

// NCHW dY -> padded GEMM B matrix [y,x,b,Cout].
void xdna_pack_conv3x3_dy_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Cout,
    int H,
    int W,
    int Kpad) {
  std::memset(dst, 0, std::size_t(Kpad) * Cout * sizeof(uint16_t));
  const int real_k = B * H * W;

#pragma omp parallel for num_threads(g_pack_threads) schedule(static)
  for (int p = 0; p < real_k; ++p) {
    const int b = p % B;
    const int pos = p / B;
    const int x = pos % W;
    const int y = pos / W;
    uint16_t* d = dst + std::size_t(p) * Cout;
    for (int co = 0; co < Cout; ++co) {
      d[co] = src[((std::size_t(b) * Cout + co) * H + y) * W + x];
    }
  }
}

}

// Faster dW packing variant. K reduction order is [Batch,Y,X] rather than
// [Y,X,Batch]. GEMM is invariant to a common K permutation, so dY uses the
// same order. The destination BO must be zero-initialized once; padding
// positions are never overwritten and therefore remain zero across steps.
extern "C" void xdna_pack_conv3x3_dw_a_bhw_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Cin,
    int H,
    int W,
    int Kpad) {
  const int real_rows = 9 * Cin;

#pragma omp parallel for num_threads(g_pack_threads) schedule(static)
  for (int row = 0; row < real_rows; ++row) {
    const int q = row / Cin;
    const int ci = row - q * Cin;
    const int ky = q / 3;
    const int kx = q - ky * 3;
    uint16_t* drow = dst + std::size_t(row) * Kpad;

    const int x0 = (kx == 0) ? 1 : 0;
    const int x1 = (kx == 2) ? W - 1 : W;
    const int ncopy = x1 - x0;
    const int src_x0 = x0 + kx - 1;

    for (int b = 0; b < B; ++b) {
      const uint16_t* plane =
          src + (std::size_t(b) * Cin + ci) * H * W;
      uint16_t* db = drow + std::size_t(b) * H * W;
      for (int y = 0; y < H; ++y) {
        const int iy = y + ky - 1;
        if (unsigned(iy) >= unsigned(H))
          continue;
        std::memcpy(
            db + std::size_t(y) * W + x0,
            plane + std::size_t(iy) * W + src_x0,
            std::size_t(ncopy) * sizeof(uint16_t));
      }
    }
  }
}

extern "C" void xdna_pack_conv3x3_dy_bhw_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Cout,
    int H,
    int W,
    int Kpad) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int p = (b * H + y) * W + x;
        uint16_t* d = dst + std::size_t(p) * Cout;
        for (int co = 0; co < Cout; ++co)
          d[co] = src[((std::size_t(b) * Cout + co) * H + y) * W + x];
      }
    }
  }
}

extern "C" void xdna_conv_set_threads(int n) {
  g_pack_threads = n > 0 ? n : 1;
}

// Pack one 128-channel slice of dX weights from original OIHW.
// Logical transformed matrix is [ky,kx,Cout,Cin]; this function selects
// Cin columns [out_start, out_start+out_count) so a fixed N=128 resident
// GEMM can produce a wider input gradient in multiple dispatches.
extern "C" void xdna_pack_conv3x3_weight_dx_slice_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int Cout,
    int Cin,
    int out_start,
    int out_count) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      for (int co = 0; co < Cout; ++co) {
        uint16_t* d =
            dst + (std::size_t((ky * 3 + kx) * Cout + co) * out_count);
        for (int j = 0; j < out_count; ++j) {
          const int ci = out_start + j;
          d[j] = src[((std::size_t(co) * Cin + ci) * 3 + (2 - ky)) * 3 +
                     (2 - kx)];
        }
      }
    }
  }
}

extern "C" void xdna_pack_conv3x3_weight_dx_slice_padded_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int Cout,
    int Cin,
    int out_start,
    int out_count,
    int dst_n) {
  const std::size_t rows = std::size_t(9) * Cout;
  std::memset(dst, 0, rows * dst_n * sizeof(uint16_t));
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      for (int co = 0; co < Cout; ++co) {
        uint16_t* d =
            dst + std::size_t((ky * 3 + kx) * Cout + co) * dst_n;
        for (int j = 0; j < out_count; ++j) {
          const int ci = out_start + j;
          d[j] = src[((std::size_t(co) * Cin + ci) * 3 + (2 - ky)) * 3 +
                     (2 - kx)];
        }
      }
    }
  }
}

extern "C" void xdna_yxbc_slice_stripe_to_channels_last_partial_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int Ctotal,
    int Cphysical,
    int Ccopy,
    int channel_offset,
    int full_h,
    int y0,
    int stripe_h,
    int W,
    int W_sched) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int ly = 0; ly < stripe_h; ++ly) {
      const int gy = y0 + ly;
      for (int x = 0; x < W; ++x) {
        const uint16_t* srcp =
            src + (((std::size_t(ly) * W_sched + x) * B + b) * Cphysical);
        uint16_t* dstp =
            dst + (((std::size_t(b) * full_h + gy) * W + x) * Ctotal +
                   channel_offset);
        std::memcpy(dstp, srcp, std::size_t(Ccopy) * sizeof(uint16_t));
      }
    }
  }
}

// ---- conv3x3w host layouts (spec: xdna_train/_designs/conv3x3w_host.py) ----

namespace {
constexpr int kGroupOutRows = 2048;
constexpr int kGroupRows = 2126;
constexpr int kKStep = 64;
constexpr int kN = 128;
}  // namespace

extern "C" {

// NCHW bf16 -> [groups][chunks][GROUP_ROWS][64].  Writes only real pixels and
// real channels (plus their halo duplicates in the previous group); the caller
// zero-fills dst once and reuses it for the same shape.
void xdna_conv3x3w_pack_input_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int H,
    int W,
    int chunks) {
  const int wp = W + 2;
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
      for (int y = 0; y < H; ++y) {
        const int cc = C - chunk * kKStep < kKStep ? C - chunk * kKStep : kKStep;
        if (cc <= 0) continue;
        const uint16_t* s =
            src + ((std::size_t(b) * C + chunk * kKStep) * H + y) * W;
        const std::size_t p0 = (std::size_t(b) * (H + 2) + y + 1) * wp + 1;
        for (int x = 0; x < W; ++x) {
          uint16_t px[kKStep];
          for (int c = 0; c < cc; ++c) px[c] = s[std::size_t(c) * H * W + x];
          const std::size_t p = p0 + x;
          const std::size_t g = p / kGroupOutRows;
          const std::size_t r = p % kGroupOutRows;
          std::memcpy(
              dst + ((g * chunks + chunk) * kGroupRows + r) * kKStep, px,
              std::size_t(cc) * sizeof(uint16_t));
          if (g > 0 && r + kGroupOutRows < std::size_t(kGroupRows))
            std::memcpy(
                dst + (((g - 1) * chunks + chunk) * kGroupRows + r +
                       kGroupOutRows) * kKStep,
                px, std::size_t(cc) * sizeof(uint16_t));
        }
      }
    }
  }
}

// OIHW bf16 -> atb-shuffled bfp16ebs8 bytes for output channels
// [128*n_block, +128).  Block order [kb][n/16][k/8][2][8 n][8 k], K order
// (chunk, dy, dx, c).  Each 8-k block = 1 exponent byte + 8 int8 mantissas.
void xdna_conv3x3w_pack_weights_bfp16(
    const uint16_t* w,
    uint8_t* dst,
    int Cout,
    int Cin,
    int chunks,
    int n_block) {
  const int nkb = chunks * 9;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int kb = 0; kb < nkb; ++kb) {
    for (int n16 = 0; n16 < kN / 16; ++n16) {
      const int chunk = kb / 9, tap = kb % 9;
      uint8_t* out = dst + std::size_t(kb * (kN / 16) + n16) * 8 * 2 * 8 * 9;
      for (int k8 = 0; k8 < 8; ++k8) {
        for (int two = 0; two < 2; ++two) {
          for (int nl = 0; nl < 8; ++nl) {
            const int n = kN * n_block + n16 * 16 + two * 8 + nl;
            uint32_t u[8];
            int mx = 0;
            for (int kl = 0; kl < 8; ++kl) {
              const int c = chunk * kKStep + k8 * 8 + kl;
              uint32_t bits = 0;
              if (c < Cin && n < Cout)
                bits = uint32_t(w[(std::size_t(n) * Cin + c) * 9 + tap]) << 16;
              u[kl] = bits;
              const int e = int((bits >> 23) & 0xFF);
              if (e > mx) mx = e;
            }
            // scale = 2^(133 - mx), exact power of two in double
            const uint64_t sb = uint64_t(1023 + 133 - mx) << 52;
            double scale;
            std::memcpy(&scale, &sb, sizeof(scale));
            *out++ = uint8_t(mx);
            for (int kl = 0; kl < 8; ++kl) {
              float f;
              std::memcpy(&f, &u[kl], sizeof(f));
              double q = __builtin_nearbyint(double(f) * scale);
              q = q < -128.0 ? -128.0 : (q > 127.0 ? 127.0 : q);
              *out++ = uint8_t(int8_t(int(q)));
            }
          }
        }
      }
    }
  }
}

// [groups*2048][128] bf16 rows -> channel slice [c_start, c_start+cout_valid)
// of an NCHW [B][C_total][H][W] tensor (valid pixels only).
void xdna_conv3x3w_unpack_output_bf16(
    const uint16_t* c,
    uint16_t* dst,
    int B,
    int H,
    int W,
    int cout_valid,
    int C_total,
    int c_start) {
  const int wp = W + 2;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b) {
    for (int y = 0; y < H; ++y) {
      const uint16_t* s = c + (std::size_t(b) * (H + 2) + y) * wp * kN;
      for (int n = 0; n < cout_valid; ++n) {
        uint16_t* d =
            dst + ((std::size_t(b) * C_total + c_start + n) * H + y) * W;
        for (int x = 0; x < W; ++x) d[x] = s[std::size_t(x) * kN + n];
      }
    }
  }
}

}  // extern "C"

// ---- conv3x3w dX weights: the forward weight w[Cout_fwd][Cin_fwd][3][3] read as
// w'[ci][co][dy][dx] = w[co][ci][2-dy][2-dx] (spatial flip + in/out swap), so the
// dX conv (Cin' = Cout_fwd, Cout' = Cin_fwd) needs no flipped copy.  Same bytes as
// xdna_conv3x3w_pack_weights_bfp16 of that tensor.
extern "C" void xdna_conv3x3w_pack_weights_dx_bfp16(
    const uint16_t* w,
    uint8_t* dst,
    int Cout_fwd,
    int Cin_fwd,
    int chunks,
    int n_block) {
  const int Cin = Cout_fwd, Cout = Cin_fwd;
  const int nkb = chunks * 9;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int kb = 0; kb < nkb; ++kb) {
    for (int n16 = 0; n16 < kN / 16; ++n16) {
      const int chunk = kb / 9, tap = kb % 9;
      uint8_t* out = dst + std::size_t(kb * (kN / 16) + n16) * 8 * 2 * 8 * 9;
      for (int k8 = 0; k8 < 8; ++k8) {
        for (int two = 0; two < 2; ++two) {
          for (int nl = 0; nl < 8; ++nl) {
            const int n = kN * n_block + n16 * 16 + two * 8 + nl;
            uint32_t u[8];
            int mx = 0;
            for (int kl = 0; kl < 8; ++kl) {
              const int c = chunk * kKStep + k8 * 8 + kl;
              uint32_t bits = 0;
              if (c < Cin && n < Cout)
                bits = uint32_t(w[(std::size_t(c) * Cin_fwd + n) * 9 + (8 - tap)]) << 16;
              u[kl] = bits;
              const int e = int((bits >> 23) & 0xFF);
              if (e > mx) mx = e;
            }
            const uint64_t sb = uint64_t(1023 + 133 - mx) << 52;
            double scale;
            std::memcpy(&scale, &sb, sizeof(scale));
            *out++ = uint8_t(mx);
            for (int kl = 0; kl < 8; ++kl) {
              float f;
              std::memcpy(&f, &u[kl], sizeof(f));
              double q = __builtin_nearbyint(double(f) * scale);
              q = q < -128.0 ? -128.0 : (q > 127.0 ? 127.0 : q);
              *out++ = uint8_t(int8_t(int(q)));
            }
          }
        }
      }
    }
  }
}
