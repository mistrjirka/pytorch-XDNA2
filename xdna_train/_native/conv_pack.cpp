#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#include <emmintrin.h>
#include <immintrin.h>
#endif

// Width of the pack loops only.  Set through num_threads() on each loop so the
// caller's OpenMP width (PyTorch's intra-op setting) is never modified.
static int g_pack_threads = 1;

extern "C" {

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

// 8x8 transpose of 16-bit elements: out[j][i] = in[i][j] (row strides in elements).
inline void transpose8x8_u16(const uint16_t* in, std::size_t is, uint16_t* out, std::size_t os) {
  __m128i r0 = _mm_loadu_si128((const __m128i*)(in + 0 * is));
  __m128i r1 = _mm_loadu_si128((const __m128i*)(in + 1 * is));
  __m128i r2 = _mm_loadu_si128((const __m128i*)(in + 2 * is));
  __m128i r3 = _mm_loadu_si128((const __m128i*)(in + 3 * is));
  __m128i r4 = _mm_loadu_si128((const __m128i*)(in + 4 * is));
  __m128i r5 = _mm_loadu_si128((const __m128i*)(in + 5 * is));
  __m128i r6 = _mm_loadu_si128((const __m128i*)(in + 6 * is));
  __m128i r7 = _mm_loadu_si128((const __m128i*)(in + 7 * is));
  __m128i a0 = _mm_unpacklo_epi16(r0, r1), a1 = _mm_unpackhi_epi16(r0, r1);
  __m128i a2 = _mm_unpacklo_epi16(r2, r3), a3 = _mm_unpackhi_epi16(r2, r3);
  __m128i a4 = _mm_unpacklo_epi16(r4, r5), a5 = _mm_unpackhi_epi16(r4, r5);
  __m128i a6 = _mm_unpacklo_epi16(r6, r7), a7 = _mm_unpackhi_epi16(r6, r7);
  __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2);
  __m128i b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3);
  __m128i b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6);
  __m128i b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
  _mm_storeu_si128((__m128i*)(out + 0 * os), _mm_unpacklo_epi64(b0, b4));
  _mm_storeu_si128((__m128i*)(out + 1 * os), _mm_unpackhi_epi64(b0, b4));
  _mm_storeu_si128((__m128i*)(out + 2 * os), _mm_unpacklo_epi64(b1, b5));
  _mm_storeu_si128((__m128i*)(out + 3 * os), _mm_unpackhi_epi64(b1, b5));
  _mm_storeu_si128((__m128i*)(out + 4 * os), _mm_unpacklo_epi64(b2, b6));
  _mm_storeu_si128((__m128i*)(out + 5 * os), _mm_unpackhi_epi64(b2, b6));
  _mm_storeu_si128((__m128i*)(out + 6 * os), _mm_unpacklo_epi64(b3, b7));
  _mm_storeu_si128((__m128i*)(out + 7 * os), _mm_unpackhi_epi64(b3, b7));
}

// tile[i][c] = s[c * hw + i] for i < n (n <= 8), c < cc.
inline void gather_pixels(const uint16_t* s, std::size_t hw, int cc, int n, uint16_t (*tile)[kKStep]) {
  int c = 0;
  if (n == 8)
    for (; c + 8 <= cc; c += 8) transpose8x8_u16(s + c * hw, hw, &tile[0][c], kKStep);
  for (; c < cc; ++c)
    for (int i = 0; i < n; ++i) tile[i][c] = s[std::size_t(c) * hw + i];
}

// d[n * dstride + x] = s[x * kN + n] for x < nx, n < nv.
inline void scatter_channels(const uint16_t* s, uint16_t* d, std::size_t dstride, int nx, int nv) {
  int x = 0;
  for (; x + 8 <= nx; x += 8) {
    int n = 0;
    for (; n + 8 <= nv; n += 8)
      transpose8x8_u16(s + std::size_t(x) * kN + n, kN, d + std::size_t(n) * dstride + x, dstride);
    for (; n < nv; ++n)
      for (int i = 0; i < 8; ++i) d[std::size_t(n) * dstride + x + i] = s[std::size_t(x + i) * kN + n];
  }
  for (; x < nx; ++x)
    for (int n = 0; n < nv; ++n) d[std::size_t(n) * dstride + x] = s[std::size_t(x) * kN + n];
}
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
        for (int xb = 0; xb < W; xb += 8) {
          const int nx = W - xb < 8 ? W - xb : 8;
          uint16_t tile[8][kKStep];
          gather_pixels(s + xb, std::size_t(H) * W, cc, nx, tile);
          for (int xi = 0; xi < nx; ++xi) {
          const int x = xb + xi;
          const uint16_t* px = tile[xi];
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
      scatter_channels(
          s, dst + ((std::size_t(b) * C_total + c_start) * H + y) * W,
          std::size_t(H) * W, W, cout_valid);
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

// ---- conv3x3w group-range variants ----

extern "C" {

// Groups [g0, g0+ng) of xdna_conv3x3w_pack_input_bf16's result, written to
// dst laid out [ng][chunks][GROUP_ROWS][64].  Same contract: only real
// pixels/channels are written, the caller zero-fills dst once per shape.
// Only the source rows that land in the range (incl. next group's halo) are read.
void xdna_conv3x3w_pack_input_range_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int H,
    int W,
    int chunks,
    int g0,
    int ng) {
  const int wp = W + 2;
  const std::size_t img = std::size_t(H + 2) * wp;
  const std::size_t plo = std::size_t(g0) * kGroupOutRows;
  const std::size_t phi =
      std::size_t(g0 + ng) * kGroupOutRows + (kGroupRows - kGroupOutRows);
  const int b0 = int(plo / img);
  const int b1 = int((phi - 1) / img) + 1 < B ? int((phi - 1) / img) + 1 : B;
  if (b1 <= b0) return;
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int b = b0; b < b1; ++b) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
      for (int y = 0; y < H; ++y) {
        const int cc = C - chunk * kKStep < kKStep ? C - chunk * kKStep : kKStep;
        const std::size_t pr = (std::size_t(b) * (H + 2) + y + 1) * wp + 1;
        if (cc <= 0 || pr + W <= plo || pr >= phi) continue;
        const int x0 = pr < plo ? int(plo - pr) : 0;
        const int x1 = pr + W > phi ? int(phi - pr) : W;
        const uint16_t* s =
            src + ((std::size_t(b) * C + chunk * kKStep) * H + y) * W;
        for (int xb = x0; xb < x1; xb += 8) {
          const int nx = x1 - xb < 8 ? x1 - xb : 8;
          uint16_t tile[8][kKStep];
          gather_pixels(s + xb, std::size_t(H) * W, cc, nx, tile);
          for (int xi = 0; xi < nx; ++xi) {
          const int x = xb + xi;
          const uint16_t* px = tile[xi];
          const std::size_t p = pr + x;
          const std::size_t g = p / kGroupOutRows;
          const std::size_t r = p % kGroupOutRows;
          if (g >= std::size_t(g0) && g < std::size_t(g0 + ng))
            std::memcpy(
                dst + (((g - g0) * chunks + chunk) * kGroupRows + r) * kKStep,
                px, std::size_t(cc) * sizeof(uint16_t));
          if (g > std::size_t(g0) && g <= std::size_t(g0 + ng) &&
              r + kGroupOutRows < std::size_t(kGroupRows))
            std::memcpy(
                dst + (((g - 1 - g0) * chunks + chunk) * kGroupRows + r +
                       kGroupOutRows) * kKStep,
                px, std::size_t(cc) * sizeof(uint16_t));
          }
        }
      }
    }
  }
}

// Rows [p0, p1) of xdna_conv3x3w_unpack_output_bf16.  `c` points at output
// row p0 (row stride 128); only valid pixels with p in [p0, p1) are written.
void xdna_conv3x3w_unpack_output_range_bf16(
    const uint16_t* c,
    uint16_t* dst,
    int B,
    int H,
    int W,
    int cout_valid,
    int C_total,
    int c_start,
    int p0,
    int p1) {
  const int wp = W + 2;
  const std::size_t img = std::size_t(H + 2) * wp;
  if (p1 <= p0) return;
  const int b0 = int(std::size_t(p0) / img);
  const int b1 = int((std::size_t(p1) - 1) / img) + 1 < B ? int((std::size_t(p1) - 1) / img) + 1 : B;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = b0; b < b1; ++b) {
    for (int y = 0; y < H; ++y) {
      const long base = long(std::size_t(b) * (H + 2) + y) * wp;
      const long xa = p0 - base > 0 ? p0 - base : 0;
      const long xb = p1 - base < W ? p1 - base : W;
      if (xb <= xa) continue;
      const uint16_t* s = c + (base + xa - p0) * kN;
      scatter_channels(
          s, dst + ((std::size_t(b) * C_total + c_start) * H + y) * W + xa,
          std::size_t(H) * W, int(xb - xa), cout_valid);
    }
  }
}


// ---- conv3x3w width tiling ----
//
// Images wider than the family limit are cut into S vertical strips of So
// output columns; each strip becomes a virtual image of Wg = So + 2 data
// columns (one real neighbour column on each side) and runs as an ordinary
// padded-grid image: only its middle So columns are valid outputs.  S = 1
// is the plain layout (Wg = W, no halo).  Virtual image vb = b * S + s covers
// source columns [s*So - 1, s*So + So + 1), zero outside [0, W).

// Groups [g0, g0+ng) of the packed input of the strip layout (see
// xdna_conv3x3w_pack_input_range_bf16 for the buffer contract).  halo_zero
// writes zeros in the two halo data columns instead of the neighbours (the
// layout dY needs for the weight gradient).
void xdna_conv3x3w_pack_input_strips_range_bf16(
    const uint16_t* src,
    uint16_t* dst,
    int B,
    int C,
    int H,
    int W,
    int S,
    int So,
    int halo_zero,
    int chunks,
    int g0,
    int ng) {
  const int Wg = S == 1 ? W : So + 2;
  const int wp = Wg + 2;
  const int Bv = B * S;
  const std::size_t img = std::size_t(H + 2) * wp;
  const std::size_t plo = std::size_t(g0) * kGroupOutRows;
  const std::size_t phi =
      std::size_t(g0 + ng) * kGroupOutRows + (kGroupRows - kGroupOutRows);
  const int v0 = int(plo / img);
  const int v1 = int((phi - 1) / img) + 1 < Bv ? int((phi - 1) / img) + 1 : Bv;
  if (v1 <= v0) return;
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int vb = v0; vb < v1; ++vb) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
      for (int y = 0; y < H; ++y) {
        const int cc = C - chunk * kKStep < kKStep ? C - chunk * kKStep : kKStep;
        const int b = vb / S, sidx = vb % S;
        const int xs0 = S == 1 ? 0 : sidx * So - 1;  // source column of data column 0
        const std::size_t pr = (std::size_t(vb) * (H + 2) + y + 1) * wp + 1;
        if (cc <= 0 || pr + Wg <= plo || pr >= phi) continue;
        // data columns [ja, jb) have a source pixel
        int ja = xs0 < 0 ? -xs0 : 0;
        int jb = W - xs0 < Wg ? W - xs0 : Wg;
        // restrict to the group range
        const int j0 = pr < plo ? int(plo - pr) : 0;
        const int j1 = pr + Wg > phi ? int(phi - pr) : Wg;
        if (ja < j0) ja = j0;
        if (jb > j1) jb = j1;
        const uint16_t* s =
            src + ((std::size_t(b) * C + chunk * kKStep) * H + y) * W + xs0;
        for (int xb = ja; xb < jb; xb += 8) {
          const int nx = jb - xb < 8 ? jb - xb : 8;
          uint16_t tile[8][kKStep];
          gather_pixels(s + xb, std::size_t(H) * W, cc, nx, tile);
          for (int xi = 0; xi < nx; ++xi) {
            const int j = xb + xi;
            if (halo_zero && S > 1 && (j == 0 || j == Wg - 1))
              std::memset(tile[xi], 0, sizeof(tile[xi]));
            const std::size_t p = pr + j;
            const std::size_t g = p / kGroupOutRows;
            const std::size_t r = p % kGroupOutRows;
            if (g >= std::size_t(g0) && g < std::size_t(g0 + ng))
              std::memcpy(
                  dst + (((g - g0) * chunks + chunk) * kGroupRows + r) * kKStep,
                  tile[xi], std::size_t(cc) * sizeof(uint16_t));
            if (g > std::size_t(g0) && g <= std::size_t(g0 + ng) &&
                r + kGroupOutRows < std::size_t(kGroupRows))
              std::memcpy(
                  dst + (((g - 1 - g0) * chunks + chunk) * kGroupRows + r +
                         kGroupOutRows) * kKStep,
                  tile[xi], std::size_t(cc) * sizeof(uint16_t));
          }
        }
      }
    }
  }
}

}  // extern "C"

namespace {
constexpr int kGroupRowsBfp = 2128;  // conv3x3w_host.GROUP_ROWS (bfp16 input layout)

// 8 bf16 values -> one bfp16ebs8 block: shared exponent byte (the largest
// biased exponent) + 8 int8 mantissas, value * 2^(133 - exp) rounded to
// nearest even and clipped (conv3x3w_host.bfp16_encode).
inline void encode_bfp_block(const uint16_t* v, uint8_t* out) {
  int mx = 0;
  for (int i = 0; i < 8; ++i) {
    const int e = (v[i] >> 7) & 0xFF;
    mx = e > mx ? e : mx;
  }
  const uint64_t sb = uint64_t(1023 + 133 - mx) << 52;
  double scale;
  std::memcpy(&scale, &sb, sizeof(scale));
  out[0] = uint8_t(mx);
  for (int i = 0; i < 8; ++i) {
    const uint32_t bits = uint32_t(v[i]) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    double q = __builtin_nearbyint(double(f) * scale);
    q = q < -128.0 ? -128.0 : (q > 127.0 ? 127.0 : q);
    out[1 + i] = uint8_t(int8_t(int(q)));
  }
}

// encode_bfp_block for 4 consecutive blocks (32 values) with AVX-512BW, in
// integers: a normal bf16 value is (128 + m) * 2^(e - 134), so its mantissa
// is (128 + m) >> (mx - e + 1) rounded to nearest even (subnormals: m >> mx).
// Bit-identical to encode_bfp_block.
__attribute__((target("avx512f,avx512bw")))
void encode_bfp_blocks4_avx512(const uint16_t* v, uint8_t exps[4], uint8_t mant[32]) {
  const __m512i x = _mm512_loadu_si512(v);
  const __m512i ff = _mm512_set1_epi16(0xFF), one = _mm512_set1_epi16(1);
  const __m512i e = _mm512_and_si512(_mm512_srli_epi16(x, 7), ff);
  // block max: each 128-bit lane is one block of 8 values
  __m512i m = _mm512_max_epu16(e, _mm512_rol_epi32(e, 16));
  m = _mm512_max_epu16(m, _mm512_shuffle_epi32(m, _MM_PERM_CDAB));
  m = _mm512_max_epu16(m, _mm512_shuffle_epi32(m, _MM_PERM_BADC));
  const __mmask32 normal = _mm512_cmpgt_epu16_mask(e, _mm512_setzero_si512());
  const __m512i m7 = _mm512_and_si512(x, _mm512_set1_epi16(0x7F));
  const __m512i v2 = _mm512_mask_blend_epi16(normal, m7, _mm512_or_si512(m7, _mm512_set1_epi16(0x80)));
  const __m512i ee = _mm512_max_epu16(e, one);
  __m512i sh = _mm512_add_epi16(_mm512_sub_epi16(m, ee), one);
  sh = _mm512_min_epu16(sh, _mm512_set1_epi16(15));
  const __m512i odd = _mm512_and_si512(_mm512_srlv_epi16(v2, sh), one);
  const __mmask32 pos = _mm512_cmpgt_epu16_mask(sh, _mm512_setzero_si512());
  const __m512i half = _mm512_sllv_epi16(one, _mm512_sub_epi16(sh, one));
  const __m512i bias = _mm512_maskz_add_epi16(pos, _mm512_sub_epi16(half, one), odd);
  __m512i q = _mm512_srlv_epi16(_mm512_add_epi16(v2, bias), sh);
  const __mmask32 neg = _mm512_test_epi16_mask(x, _mm512_set1_epi16(int16_t(0x8000)));
  q = _mm512_min_epu16(q, _mm512_mask_blend_epi16(neg, _mm512_set1_epi16(127), _mm512_set1_epi16(128)));
  q = _mm512_mask_sub_epi16(q, neg, _mm512_setzero_si512(), q);
  _mm256_storeu_si256((__m256i*)mant, _mm512_cvtepi16_epi8(q));
  alignas(64) uint16_t mm[32];
  _mm512_store_si512(mm, m);
  for (int k = 0; k < 4; ++k) exps[k] = uint8_t(mm[8 * k]);
}

bool have_avx512bw() {
  static const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
  return ok;
}

// One pixel's nblk (<= 8) channel blocks -> enc[k] = 9 bytes each.
inline void encode_pixel_bfp(const uint16_t* px, int nblk, uint8_t (*enc)[9]) {
  if (have_avx512bw()) {
    for (int k0 = 0; k0 < nblk; k0 += 4) {
      uint8_t exps[4], mant[32];
      encode_bfp_blocks4_avx512(px + 8 * k0, exps, mant);
      for (int k = k0; k < nblk && k < k0 + 4; ++k) {
        enc[k][0] = exps[k - k0];
        std::memcpy(&enc[k][1], mant + 8 * (k - k0), 8);
      }
    }
    return;
  }
  for (int k = 0; k < nblk; ++k) encode_bfp_block(px + 8 * k, enc[k]);
}
}  // namespace

extern "C" {

// xdna_conv3x3w_pack_input_strips_range_bf16 for the forward/dX kernel's
// bfp16 input: dst laid out [ng][chunks][8 channel blocks][GROUP_ROWS][9 B],
// GROUP_ROWS = 2128, each block = one pixel's 8 channels.  Same contract:
// only blocks holding real pixels are written, the caller zero-fills dst once
// per shape (zero bytes are a valid zero block).  dst16, if not null, also
// receives xdna_conv3x3w_pack_input_strips_range_bf16's result for the same
// groups (the layout of the deferred weight gradient), from the same read.
void xdna_conv3x3w_pack_input_strips_range_bfp16(
    const uint16_t* src,
    uint8_t* dst,
    uint16_t* dst16,
    int B,
    int C,
    int H,
    int W,
    int S,
    int So,
    int halo_zero,
    int chunks,
    int g0,
    int ng) {
  const int Wg = S == 1 ? W : So + 2;
  const int wp = Wg + 2;
  const int Bv = B * S;
  const std::size_t img = std::size_t(H + 2) * wp;
  const std::size_t plo = std::size_t(g0) * kGroupOutRows;
  const std::size_t phi =
      std::size_t(g0 + ng) * kGroupOutRows + (kGroupRowsBfp - kGroupOutRows);
  const int v0 = int(plo / img);
  const int v1 = int((phi - 1) / img) + 1 < Bv ? int((phi - 1) / img) + 1 : Bv;
  if (v1 <= v0) return;
  const std::size_t plane = std::size_t(kGroupRowsBfp) * 9;  // one channel block
#pragma omp parallel for num_threads(g_pack_threads) collapse(3) schedule(static)
  for (int vb = v0; vb < v1; ++vb) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
      for (int y = 0; y < H; ++y) {
        const int cc = C - chunk * kKStep < kKStep ? C - chunk * kKStep : kKStep;
        const int nblk = (cc + 7) / 8;
        const int b = vb / S, sidx = vb % S;
        const int xs0 = S == 1 ? 0 : sidx * So - 1;
        const std::size_t pr = (std::size_t(vb) * (H + 2) + y + 1) * wp + 1;
        if (cc <= 0 || pr + Wg <= plo || pr >= phi) continue;
        int ja = xs0 < 0 ? -xs0 : 0;
        int jb = W - xs0 < Wg ? W - xs0 : Wg;
        const int j0 = pr < plo ? int(plo - pr) : 0;
        const int j1 = pr + Wg > phi ? int(phi - pr) : Wg;
        if (ja < j0) ja = j0;
        if (jb > j1) jb = j1;
        const uint16_t* s =
            src + ((std::size_t(b) * C + chunk * kKStep) * H + y) * W + xs0;
        for (int xb = ja; xb < jb; xb += 8) {
          const int nx = jb - xb < 8 ? jb - xb : 8;
          uint16_t tile[8][kKStep];
          if (cc < kKStep) std::memset(tile, 0, sizeof(tile));
          gather_pixels(s + xb, std::size_t(H) * W, cc, nx, tile);
          for (int xi = 0; xi < nx; ++xi) {
            const int j = xb + xi;
            if (halo_zero && S > 1 && (j == 0 || j == Wg - 1))
              std::memset(tile[xi], 0, sizeof(tile[xi]));
            uint8_t enc[8][9];
            encode_pixel_bfp(tile[xi], nblk, enc);
            const std::size_t p = pr + j;
            const std::size_t g = p / kGroupOutRows;
            const std::size_t r = p % kGroupOutRows;
            if (dst16) {
              if (g >= std::size_t(g0) && g < std::size_t(g0 + ng))
                std::memcpy(
                    dst16 + (((g - g0) * chunks + chunk) * kGroupRows + r) * kKStep,
                    tile[xi], std::size_t(cc) * sizeof(uint16_t));
              if (g > std::size_t(g0) && g <= std::size_t(g0 + ng) &&
                  r + kGroupOutRows < std::size_t(kGroupRows))
                std::memcpy(
                    dst16 + (((g - 1 - g0) * chunks + chunk) * kGroupRows + r +
                             kGroupOutRows) * kKStep,
                    tile[xi], std::size_t(cc) * sizeof(uint16_t));
            }
            if (g >= std::size_t(g0) && g < std::size_t(g0 + ng)) {
              uint8_t* d = dst + ((g - g0) * chunks + chunk) * 8 * plane + r * 9;
              for (int k = 0; k < nblk; ++k) std::memcpy(d + k * plane, enc[k], 9);
            }
            if (g > std::size_t(g0) && g <= std::size_t(g0 + ng) &&
                r + kGroupOutRows < std::size_t(kGroupRowsBfp)) {
              uint8_t* d = dst + ((g - 1 - g0) * chunks + chunk) * 8 * plane +
                           (r + kGroupOutRows) * 9;
              for (int k = 0; k < nblk; ++k) std::memcpy(d + k * plane, enc[k], 9);
            }
          }
        }
      }
    }
  }
}

// Rows [p0, p1) of the strip layout's output (rows of 128 channels, `c`
// points at row p0) -> channel slice [c_start, c_start + cout_valid) of an
// NCHW [B][C_total][H][W] tensor.  Strip s's output column t sits at grid row
// base + 1 + t and lands at image column s*So + t.
void xdna_conv3x3w_unpack_output_strips_range_bf16(
    const uint16_t* c,
    uint16_t* dst,
    int B,
    int H,
    int W,
    int S,
    int So,
    int cout_valid,
    int C_total,
    int c_start,
    int p0,
    int p1) {
  const int Wg = S == 1 ? W : So + 2;
  const int wp = Wg + 2;
  const int Bv = B * S;
  const std::size_t img = std::size_t(H + 2) * wp;
  if (p1 <= p0) return;
  const int v0 = int(std::size_t(p0) / img);
  const int v1 = int((std::size_t(p1) - 1) / img) + 1 < Bv ? int((std::size_t(p1) - 1) / img) + 1 : Bv;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int vb = v0; vb < v1; ++vb) {
    for (int y = 0; y < H; ++y) {
      const int b = vb / S, sidx = vb % S;
      const long base = long(std::size_t(vb) * (H + 2) + y) * wp;
      const long first = S == 1 ? 0 : 1;           // grid offset of output column 0
      const int cols = S == 1 ? W : (W - sidx * So < So ? W - sidx * So : So);
      const long ta = p0 - (base + first) > 0 ? p0 - (base + first) : 0;
      const long tb = p1 - (base + first) < cols ? p1 - (base + first) : cols;
      if (tb <= ta) continue;
      const int x0 = (S == 1 ? 0 : sidx * So);
      const uint16_t* s = c + (base + first + ta - p0) * kN;
      scatter_channels(
          s, dst + ((std::size_t(b) * C_total + c_start) * H + y) * W + x0 + ta,
          std::size_t(H) * W, int(tb - ta), cout_valid);
    }
  }
}

}  // extern "C"
