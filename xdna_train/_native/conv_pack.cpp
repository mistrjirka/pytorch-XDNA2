#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "conv_pack.h"

#ifdef _OPENMP
#include <omp.h>
#include <emmintrin.h>
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

// Branch-free so the loops below vectorize (`omp simd`): these run on every
// element of a lazy value, as often as ATen's own vectorized loops would.
[[gnu::always_inline]] inline float bf16_to_f32(uint16_t v) { return __builtin_bit_cast(float, uint32_t(v) << 16); }

// Round to nearest even, NaN -> canonical quiet NaN (c10::BFloat16's rounding).
[[gnu::always_inline]] inline uint16_t f32_to_bf16(float f) {
  const uint32_t u = __builtin_bit_cast(uint32_t, f);
  const uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
  const uint32_t nan = 0u - uint32_t((u & 0x7fffffffu) > 0x7f800000u);  // all ones for NaN
  return uint16_t((r & ~nan) | (0x7fc0u & nan));
}

// ATen's CPU leaky_relu for bf16: in fp32, x > 0 ? x : x * slope (NaN stays NaN).
// Written as a multiply by a selected factor, which is the same for every x.
// The factor is a bit blend, not a branch, so the loops vectorize.
[[gnu::always_inline]] inline float leaky_factor(float x, float slope) {
  const uint32_t m = 0u - uint32_t(x > 0.f);
  return __builtin_bit_cast(float, (0x3f800000u & m) | (__builtin_bit_cast(uint32_t, slope) & ~m));
}
[[gnu::always_inline]] inline float leaky(float x, float slope) { return x * leaky_factor(x, slope); }

// ATen's CPU relu is clamp_min(x, 0), vectorized as max(0, x) = 0 > x ? 0 : x:
// NaN and -0 pass through, -inf becomes 0.  LeakyReLU with slope 0 differs
// (-inf * 0 is NaN), so ReLU is its own activation.
[[gnu::always_inline]] inline float relu(float x) {
  const uint32_t keep = 0u - uint32_t(!(x < 0.f));
  return __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, x) & keep);
}

// Activation codes shared with torch_xdna_device.cpp (XdnaAct).
constexpr int kActNone = 0, kActLeaky = 1, kActRelu = 2;

// A lazy per-channel pointwise value, y = act(x * scale + shift).  The pack,
// the materialize kernel and the LeakyReLU backward share pointwise_one, so
// every consumer of a lazy value sees the same bytes.
struct Pointwise {
  const float* scale;  // per channel, nullptr: no affine
  const float* shift;
  int act;             // kActNone, kActLeaky (with slope) or kActRelu
  float slope;
};

template <bool Affine, int Act>
[[gnu::always_inline]] inline uint16_t pointwise_one(uint16_t v, float scale, float shift, float slope) {
  float x = bf16_to_f32(v);
  // Eager BatchNorm writes a bf16 tensor before the activation reads it, so
  // round between the two steps to match it.
  if constexpr (Affine) x = bf16_to_f32(f32_to_bf16(x * scale + shift));
  if constexpr (Act == kActLeaky) x = leaky(x, slope);
  if constexpr (Act == kActRelu) x = relu(x);
  return f32_to_bf16(x);
}

// v[i] = value of v[i] in channel c0 + i * cstep (cstep 1: a pixel's channels,
// 0: one channel's plane), i < n.  src may equal dst.
template <bool Affine, int Act>
[[gnu::always_inline]] inline void pointwise_span(const uint16_t* src, uint16_t* dst, long n, const Pointwise& pw, int c0,
                           int cstep) {
  if (cstep == 0) {
    const float sc = Affine ? pw.scale[c0] : 0.f, sh = Affine ? pw.shift[c0] : 0.f;
#pragma omp simd
    for (long i = 0; i < n; ++i) dst[i] = pointwise_one<Affine, Act>(src[i], sc, sh, pw.slope);
  } else {
    const float* sc = Affine ? pw.scale + c0 : nullptr;
    const float* sh = Affine ? pw.shift + c0 : nullptr;
#pragma omp simd
    for (long i = 0; i < n; ++i)
      dst[i] = pointwise_one<Affine, Act>(src[i], Affine ? sc[i] : 0.f, Affine ? sh[i] : 0.f, pw.slope);
  }
}

// The extension builds for baseline x86-64 (SSE2, 4 floats per instruction),
// while ATen's pointwise kernels dispatch to AVX2/AVX-512 at run time.  The
// hot lazy-value loops get the same: one clone per level, picked at load time.
// The helpers they call are always_inline: inside the large cloned function
// GCC otherwise stopped inlining them and the affine+activation loops ran at
// baseline speed (7.5 ms vs 2.1 ms for 8M elements on one core).
// No FMA contraction (-std=c++20 implies -ffp-contract=off): a*b+c rounds
// twice, as in the eager kernels these results must match bit for bit.
#define XDNA_SIMD_CLONES \
  __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))

XDNA_SIMD_CLONES
void apply_pointwise(const uint16_t* src, uint16_t* dst, long n, const Pointwise& pw, int c0,
                     int cstep) {
  if (pw.scale) {
    if (pw.act == kActLeaky) pointwise_span<true, kActLeaky>(src, dst, n, pw, c0, cstep);
    else if (pw.act == kActRelu) pointwise_span<true, kActRelu>(src, dst, n, pw, c0, cstep);
    else pointwise_span<true, kActNone>(src, dst, n, pw, c0, cstep);
  } else if (pw.act == kActLeaky) {
    pointwise_span<false, kActLeaky>(src, dst, n, pw, c0, cstep);
  } else if (pw.act == kActRelu) {
    pointwise_span<false, kActRelu>(src, dst, n, pw, c0, cstep);
  } else if (src != dst) {
    std::memcpy(dst, src, std::size_t(n) * sizeof(uint16_t));
  }
}

XDNA_SIMD_CLONES
void leaky_backward_plane(const uint16_t* dy, const uint16_t* src, uint16_t* dx, long n, float sc,
                          float sh, float slope) {
#pragma omp simd
  for (long i = 0; i < n; ++i) {
    const float z = bf16_to_f32(pointwise_one<true, kActNone>(src[i], sc, sh, 0.f));
    dx[i] = f32_to_bf16(bf16_to_f32(dy[i]) * leaky_factor(z, slope));
  }
}

XDNA_SIMD_CLONES
void relu_backward_plane(const uint16_t* dy, const uint16_t* src, uint16_t* dx, long n,
                         const float* sc, const float* sh) {
  auto keep = [](uint16_t z) { return uint16_t(0u - uint32_t(!(bf16_to_f32(z) <= 0.f))); };
  if (sc) {
    const float a = *sc, b = *sh;
#pragma omp simd
    for (long i = 0; i < n; ++i) dx[i] = dy[i] & keep(pointwise_one<true, kActNone>(src[i], a, b, 0.f));
  } else {
#pragma omp simd
    for (long i = 0; i < n; ++i) dx[i] = dy[i] & keep(src[i]);
  }
}

XDNA_SIMD_CLONES
void add_bf16_span(const uint16_t* a, const uint16_t* b, uint16_t* out, long n) {
#pragma omp simd
  for (long i = 0; i < n; ++i) out[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

// sum[n] += x, sumsq[n] += x * x over nrows rows of kN channels (n < nv), in
// double: exact enough that BatchNorm's bf16-rounded mean and variance come
// out as from the region kernel's two-level sum.
// One pixel's channels [c0, c0 + n) of the rows backward kernels.
XDNA_SIMD_CLONES
void act_backward_span(const uint16_t* dy, const uint16_t* x, uint16_t* dx, int n,
                       const float* scale, const float* shift, int mode, float slope) {
  if (mode == 1) {
    if (scale) {
#pragma omp simd
      for (int i = 0; i < n; ++i) {
        const float z = bf16_to_f32(pointwise_one<true, kActNone>(x[i], scale[i], shift[i], 0.f));
        dx[i] = f32_to_bf16(bf16_to_f32(dy[i]) * leaky_factor(z, slope));
      }
    } else {
#pragma omp simd
      for (int i = 0; i < n; ++i)
        dx[i] = f32_to_bf16(bf16_to_f32(dy[i]) * leaky_factor(bf16_to_f32(x[i]), slope));
    }
  } else {
    auto keep = [](uint16_t z) { return uint16_t(0u - uint32_t(!(bf16_to_f32(z) <= 0.f))); };
    if (scale) {
#pragma omp simd
      for (int i = 0; i < n; ++i)
        dx[i] = dy[i] & keep(pointwise_one<true, kActNone>(x[i], scale[i], shift[i], 0.f));
    } else {
#pragma omp simd
      for (int i = 0; i < n; ++i) dx[i] = dy[i] & keep(x[i]);
    }
  }
}

XDNA_SIMD_CLONES
void bn_backward_sums_span(const uint16_t* dy, const uint16_t* x, int n, const float* mean,
                           double* sum, double* dotp) {
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    const double g = bf16_to_f32(dy[i]);
    sum[i] += g;
    dotp[i] += (double(bf16_to_f32(x[i])) - double(mean[i])) * g;
  }
}

XDNA_SIMD_CLONES
void bn_backward_dx_span(const uint16_t* dy, const uint16_t* x, uint16_t* dx, int n,
                         const float* mean, const float* k, const float* grad_mean,
                         const float* invstd, const float* w) {
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    const float t = (bf16_to_f32(x[i]) - mean[i]) * k[i];
    dx[i] = f32_to_bf16((bf16_to_f32(dy[i]) - grad_mean[i] - t) * invstd[i] * w[i]);
  }
}

XDNA_SIMD_CLONES
void sum4_span(const uint16_t* a, const uint16_t* b, const uint16_t* c, const uint16_t* d,
               uint16_t* out, int n) {
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    float acc = 0.f;
    acc += bf16_to_f32(a[i]);
    acc += bf16_to_f32(b[i]);
    acc += bf16_to_f32(c[i]);
    acc += bf16_to_f32(d[i]);
    out[i] = f32_to_bf16(acc);
  }
}

XDNA_SIMD_CLONES
void accumulate_channel_stats(const uint16_t* rows, long nrows, int nv, double* sum,
                              double* sumsq) {
  for (long r = 0; r < nrows; ++r) {
    const uint16_t* row = rows + r * kN;
#pragma omp simd
    for (int n = 0; n < nv; ++n) {
      const double v = bf16_to_f32(row[n]);
      sum[n] += v;
      sumsq[n] += v * v;
    }
  }
}

// tile[xi][off + k] = channel k0 + k of segment sg at target pixel (b, y, x0 + xi),
// for xi < nx (<= 8), k < n, with the segment's pointwise applied.
inline void gather_segment(const XdnaPackSegment& sg, int b, int y, int x0, int nx, int k0, int n,
                           int H, int W, uint16_t (*tile)[kKStep], int off) {
  const int up = sg.up;
  const int Hs = H / up, Ws = W / up;
  auto* t = reinterpret_cast<uint16_t (*)[kKStep]>(&tile[0][off]);
  if (sg.rows) {
    // Channels are contiguous in a row: copies, no transpose.
    const int sy = y / up;
    for (int xi = 0; xi < nx; ++xi) {
      const int sx = (x0 + xi) / up;
      std::size_t row;
      if (sg.rows_s == 1) {
        row = (std::size_t(b) * (Hs + 2) + sy) * sg.rows_wp + sx;
      } else {
        const int strip = sx / sg.rows_so;
        row = ((std::size_t(b) * sg.rows_s + strip) * (Hs + 2) + sy) * sg.rows_wp + 1 +
              (sx - strip * sg.rows_so);
      }
      for (int k = k0; k < k0 + n;) {
        const int blk = k / kN, col = k % kN;
        const int cnt = kN - col < k0 + n - k ? kN - col : k0 + n - k;
        std::memcpy(t[xi] + (k - k0), sg.rows + blk * sg.block_stride + row * kN + col,
                    std::size_t(cnt) * sizeof(uint16_t));
        k += cnt;
      }
    }
  } else if (up == 1) {
    const std::size_t hw = std::size_t(Hs) * Ws;
    const uint16_t* s = sg.src + ((std::size_t(b) * sg.channels + k0) * Hs + y / up) * Ws;
    gather_pixels(s + x0, hw, n, nx, t);
  } else {
    const std::size_t hw = std::size_t(Hs) * Ws;
    const uint16_t* s = sg.src + ((std::size_t(b) * sg.channels + k0) * Hs + y / up) * Ws;
    // The source columns under the nx target columns, gathered once.
    const int sa = x0 / up, sb = (x0 + nx - 1) / up + 1;
    uint16_t src_tile[8][kKStep];
    gather_pixels(s + sa, hw, n, sb - sa, src_tile);
    for (int xi = 0; xi < nx; ++xi)
      std::memcpy(t[xi], src_tile[(x0 + xi) / up - sa], std::size_t(n) * sizeof(uint16_t));
  }
  if (sg.scale || sg.act) {
    const Pointwise pw{sg.scale, sg.shift, sg.act, sg.slope};
    for (int xi = 0; xi < nx; ++xi) apply_pointwise(t[xi], t[xi], n, pw, k0, 1);
  }
}

// Grid row of image pixel (b, y, x) of grid g.
inline std::size_t row_of(const XdnaRowsGeom& g, int b, int y, int x) {
  if (g.S == 1) return (std::size_t(b) * (g.H + 2) + y) * g.wp + x;
  const int s = x / g.So;
  return ((std::size_t(b) * g.S + s) * (g.H + 2) + y) * g.wp + 1 + (x - s * g.So);
}

// f(row, b, y, x0, cols) for every run of valid output pixels of grid g: grid
// rows [row, row + cols) hold image pixels (b, y, x0 .. x0 + cols).
template <class F>
void for_valid_rows(const XdnaRowsGeom& g, F&& f) {
  const int Bv = g.B * g.S;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int vb = 0; vb < Bv; ++vb) {
    for (int y = 0; y < g.H; ++y) {
      const int b = vb / g.S, sidx = vb % g.S;
      const long base = long(std::size_t(vb) * (g.H + 2) + y) * g.wp;
      const long first = g.S == 1 ? 0 : 1;
      const int cols = g.S == 1 ? g.W : (g.W - sidx * g.So < g.So ? g.W - sidx * g.So : g.So);
      if (cols > 0) f(base + first, b, y, g.S == 1 ? 0 : sidx * g.So, cols);
    }
  }
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
// xdna_conv3x3w_pack_input_range_bf16 for the buffer contract), for the value
// the segments describe (conv_pack.h).  halo_zero writes zeros in the two halo
// data columns instead of the neighbours (the layout dY needs for the weight
// gradient).  A segment's pointwise applies to every real pixel while packing
// it; the zero padding stays zero.
void xdna_conv3x3w_pack_segments_bf16(
    const XdnaPackSegment* segs,
    int nsegs,
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
        const int cbase = chunk * kKStep;
        const int cc = C - cbase < kKStep ? C - cbase : kKStep;
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
        for (int xb = ja; xb < jb; xb += 8) {
          const int nx = jb - xb < 8 ? jb - xb : 8;
          uint16_t tile[8][kKStep];
          for (int si = 0; si < nsegs; ++si) {
            const XdnaPackSegment& sg = segs[si];
            const int lo = sg.c0 > cbase ? sg.c0 : cbase;
            const int hi = sg.c0 + sg.channels < cbase + cc ? sg.c0 + sg.channels : cbase + cc;
            if (lo < hi) gather_segment(sg, b, y, xs0 + xb, nx, lo - sg.c0, hi - lo, H, W, tile, lo - cbase);
          }
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

void xdna_conv3x3w_pack_input_strips_range_bf16(
    const uint16_t* src, uint16_t* dst, int B, int C, int H, int W, int S, int So,
    int halo_zero, int chunks, int g0, int ng) {
  const XdnaPackSegment seg{src, 0, C, 1, nullptr, nullptr, kActNone, 0.f, nullptr, 0, 0, 0, 0};
  xdna_conv3x3w_pack_segments_bf16(&seg, 1, dst, B, C, H, W, S, So, halo_zero, chunks, g0, ng);
}

// With the same math as the pack above, so every reader sees the same bytes.
void xdna_lazy_segments_write_bf16(
    const XdnaPackSegment* segs, int nsegs, uint16_t* dst, int B, int C, int H, int W) {
  const std::size_t hw = std::size_t(H) * W;
  for (int si = 0; si < nsegs; ++si) {
    const XdnaPackSegment& sg = segs[si];
    const Pointwise pw{sg.scale, sg.shift, sg.act, sg.slope};
    const int up = sg.up;
    const int Hs = H / up, Ws = W / up;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
    for (int b = 0; b < B; ++b)
      for (int k = 0; k < sg.channels; ++k) {
        const uint16_t* sp = sg.src + (std::size_t(b) * sg.channels + k) * Hs * Ws;
        uint16_t* dp = dst + (std::size_t(b) * C + sg.c0 + k) * hw;
        if (up == 1) {
          apply_pointwise(sp, dp, long(hw), pw, k, 0);
          continue;
        }
        for (int y = 0; y < H; ++y) {
          const uint16_t* srow = sp + std::size_t(y / up) * Ws;
          uint16_t* drow = dp + std::size_t(y) * W;
          for (int x = 0; x < W; ++x) drow[x] = srow[x / up];
          apply_pointwise(drow, drow, W, pw, k, 0);
        }
      }
  }
}

// dx = leaky_relu_backward(dy, z) where z is the lazy affine value of src (no
// activation), computed as the pack computes it, so z is never written.  All
// contiguous NCHW bf16.  ATen: in fp32, z > 0 ? dy : dy * slope.
void xdna_leaky_backward_pw_bf16(
    const uint16_t* dy, const uint16_t* src, uint16_t* dx, int B, int C, long HW,
    const float* scale, const float* shift, float slope) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c) {
      const std::size_t off = (std::size_t(b) * C + c) * std::size_t(HW);
      leaky_backward_plane(dy + off, src + off, dx + off, HW, scale[c], shift[c], slope);
    }
}

// dx = threshold_backward(dy, relu(z), 0) where z is the lazy value's input
// (src, or its affine when scale is set), so relu(z) is never written.
// ATen: relu(z) <= 0 ? 0 : dy, which keeps dy where z > 0 or z is NaN.  A
// select, not a multiply: an infinite dy times 0 would be NaN.
void xdna_relu_backward_pw_bf16(
    const uint16_t* dy, const uint16_t* src, uint16_t* dx, int B, int C, long HW,
    const float* scale, const float* shift) {
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c) {
      const std::size_t off = (std::size_t(b) * C + c) * std::size_t(HW);
      relu_backward_plane(dy + off, src + off, dx + off, HW, scale ? scale + c : nullptr,
                          scale ? shift + c : nullptr);
    }
}

// out = a + b for contiguous NCHW bf16, where a and b may each be a lazy
// pointwise value of its src (pw_a / pw_b with no scale and no act: the bytes
// themselves).  ATen's bf16 add with alpha 1: fp32 sum of the two bf16 values,
// rounded once.  Per plane in L1-sized chunks, so each step vectorizes.
void xdna_add_pw_bf16(
    const uint16_t* a, const float* a_scale, const float* a_shift, int a_act, float a_slope,
    const uint16_t* b, const float* b_scale, const float* b_shift, int b_act, float b_slope,
    uint16_t* out, int B, int C, long HW) {
  const Pointwise pa{a_scale, a_shift, a_act, a_slope};
  const Pointwise pb{b_scale, b_shift, b_act, b_slope};
  constexpr long kChunk = 2048;
#pragma omp parallel for num_threads(g_pack_threads) collapse(2) schedule(static)
  for (int n = 0; n < B; ++n)
    for (int c = 0; c < C; ++c) {
      const std::size_t off = (std::size_t(n) * C + c) * std::size_t(HW);
      alignas(64) uint16_t ta[kChunk], tb[kChunk];
      for (long i0 = 0; i0 < HW; i0 += kChunk) {
        const long m = HW - i0 < kChunk ? HW - i0 : kChunk;
        const uint16_t* va = a + off + i0;
        const uint16_t* vb = b + off + i0;
        if (pa.scale || pa.act) { apply_pointwise(va, ta, m, pa, c, 0); va = ta; }
        if (pb.scale || pb.act) { apply_pointwise(vb, tb, m, pb, c, 0); vb = tb; }
        add_bf16_span(va, vb, out + off + i0, m);
      }
    }
}

// Rows [p0, p1) of the strip layout's output (rows of 128 channels, `c`
// points at row p0) -> channel slice [c_start, c_start + cout_valid) of an
// NCHW [B][C_total][H][W] tensor.  Strip s's output column t sits at grid row
// base + 1 + t and lands at image column s*So + t.
// As xdna_conv3x3w_unpack_output_strips_range_bf16, and when sum / sumsq are
// set, adds each valid output's value and square per channel to them
// (cout_valid entries each) while the rows are in cache: the statistics a
// following BatchNorm needs (stage 3 of docs/fusion-analysis.md).  dst may be
// null: statistics only.
void xdna_conv3x3w_unpack_output_strips_range_stats_bf16(
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
    int p1,
    double* sum,
    double* sumsq) {
  const int Wg = S == 1 ? W : So + 2;
  const int wp = Wg + 2;
  const int Bv = B * S;
  const std::size_t img = std::size_t(H + 2) * wp;
  if (p1 <= p0) return;
  const int v0 = int(std::size_t(p0) / img);
  const int v1 = int((std::size_t(p1) - 1) / img) + 1 < Bv ? int((std::size_t(p1) - 1) / img) + 1 : Bv;
#pragma omp parallel num_threads(g_pack_threads)
  {
    double local[2 * kN] = {};
#pragma omp for collapse(2) schedule(static)
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
        if (sum) accumulate_channel_stats(s, tb - ta, cout_valid, local, local + kN);
        if (dst)
          scatter_channels(
              s, dst + ((std::size_t(b) * C_total + c_start) * H + y) * W + x0 + ta,
              std::size_t(H) * W, int(tb - ta), cout_valid);
      }
    }
    if (sum) {
#pragma omp critical(xdna_unpack_stats)
      for (int n = 0; n < cout_valid; ++n) {
        sum[n] += local[n];
        sumsq[n] += local[kN + n];
      }
    }
  }
}

void xdna_conv3x3w_unpack_output_strips_range_bf16(
    const uint16_t* c, uint16_t* dst, int B, int H, int W, int S, int So, int cout_valid,
    int C_total, int c_start, int p0, int p1) {
  xdna_conv3x3w_unpack_output_strips_range_stats_bf16(
      c, dst, B, H, W, S, So, cout_valid, C_total, c_start, p0, p1, nullptr, nullptr);
}

void xdna_rows_act_backward_bf16(
    const XdnaRowsGeom* g, const uint16_t* dy, const uint16_t* x, uint16_t* dx,
    const float* scale, const float* shift, int mode, float slope) {
  for_valid_rows(*g, [&](long row, int, int, int, int cols) {
    for (int nb = 0; nb * kN < g->C; ++nb) {
      const int nv = g->C - nb * kN < kN ? g->C - nb * kN : kN;
      const std::size_t off = std::size_t(nb) * g->block_stride + std::size_t(row) * kN;
      for (int t = 0; t < cols; ++t)
        act_backward_span(dy + off + t * kN, x + off + t * kN, dx + off + t * kN, nv,
                          scale ? scale + nb * kN : nullptr, shift ? shift + nb * kN : nullptr,
                          mode, slope);
    }
  });
}

void xdna_rows_bn_backward_bf16(
    const XdnaRowsGeom* g, const uint16_t* dy, const uint16_t* x, uint16_t* dx,
    const float* mean, const float* invstd, const float* w, float* dgamma, float* dbeta) {
  const int C = g->C;
  std::vector<double> sum(C, 0.0), dotp(C, 0.0);
  const int Bv = g->B * g->S;
#pragma omp parallel num_threads(g_pack_threads)
  {
    std::vector<double> ls(C, 0.0), ld(C, 0.0);
#pragma omp for collapse(2) schedule(static)
    for (int vb = 0; vb < Bv; ++vb) {
      for (int y = 0; y < g->H; ++y) {
        const int sidx = vb % g->S;
        const long row = long(std::size_t(vb) * (g->H + 2) + y) * g->wp + (g->S == 1 ? 0 : 1);
        const int cols = g->S == 1 ? g->W : (g->W - sidx * g->So < g->So ? g->W - sidx * g->So : g->So);
        for (int nb = 0; nb * kN < C; ++nb) {
          const int nv = C - nb * kN < kN ? C - nb * kN : kN;
          const std::size_t off = std::size_t(nb) * g->block_stride + std::size_t(row) * kN;
          for (int t = 0; t < cols; ++t)
            bn_backward_sums_span(dy + off + t * kN, x + off + t * kN, nv, mean + nb * kN,
                                  ls.data() + nb * kN, ld.data() + nb * kN);
        }
      }
    }
#pragma omp critical(xdna_rows_bn_backward)
    for (int c = 0; c < C; ++c) {
      sum[c] += ls[c];
      dotp[c] += ld[c];
    }
  }
  const double n = double(g->B) * g->H * g->W;
  std::vector<float> k(C), grad_mean(C), ones(C, 1.0f);
  for (int c = 0; c < C; ++c) {
    const float fs = float(sum[c]), fd = float(dotp[c]);
    k[c] = float(fd * invstd[c] * invstd[c] / float(n));
    grad_mean[c] = float(fs / float(n));
    if (dgamma) dgamma[c] = fd * invstd[c];
    if (dbeta) dbeta[c] = fs;
  }
  if (!dx) return;
  const float* wv = w ? w : ones.data();
  for_valid_rows(*g, [&](long row, int, int, int, int cols) {
    for (int nb = 0; nb * kN < C; ++nb) {
      const int nv = C - nb * kN < kN ? C - nb * kN : kN;
      const std::size_t off = std::size_t(nb) * g->block_stride + std::size_t(row) * kN;
      for (int t = 0; t < cols; ++t)
        bn_backward_dx_span(dy + off + t * kN, x + off + t * kN, dx + off + t * kN, nv,
                            mean + nb * kN, k.data() + nb * kN, grad_mean.data() + nb * kN,
                            invstd + nb * kN, wv + nb * kN);
    }
  });
}

void xdna_rows_add_bf16(
    const XdnaRowsGeom* g, const uint16_t* a, const float* a_scale, const float* a_shift,
    int a_act, float a_slope, const uint16_t* b, const float* b_scale, const float* b_shift,
    int b_act, float b_slope, uint16_t* out) {
  const Pointwise pa{a_scale, a_shift, a_act, a_slope};
  const Pointwise pb{b_scale, b_shift, b_act, b_slope};
  for_valid_rows(*g, [&](long row, int, int, int, int cols) {
    uint16_t ta[kN], tb[kN];
    for (int nb = 0; nb * kN < g->C; ++nb) {
      const int nv = g->C - nb * kN < kN ? g->C - nb * kN : kN;
      const std::size_t off = std::size_t(nb) * g->block_stride + std::size_t(row) * kN;
      for (int t = 0; t < cols; ++t) {
        const uint16_t* va = a + off + t * kN;
        const uint16_t* vb = b + off + t * kN;
        if (pa.scale || pa.act) { apply_pointwise(va, ta, nv, pa, nb * kN, 1); va = ta; }
        if (pb.scale || pb.act) { apply_pointwise(vb, tb, nv, pb, nb * kN, 1); vb = tb; }
        add_bf16_span(va, vb, out + off + t * kN, nv);
      }
    }
  });
}

void xdna_rows_upsample2x_backward_bf16(
    const XdnaRowsGeom* gb, const uint16_t* go, const XdnaRowsGeom* gs, uint16_t* gi) {
  for_valid_rows(*gs, [&](long row, int b, int y, int x0, int cols) {
    for (int t = 0; t < cols; ++t) {
      const int x = x0 + t;
      const std::size_t r00 = row_of(*gb, b, 2 * y, 2 * x), r01 = row_of(*gb, b, 2 * y, 2 * x + 1);
      const std::size_t r10 = row_of(*gb, b, 2 * y + 1, 2 * x);
      const std::size_t r11 = row_of(*gb, b, 2 * y + 1, 2 * x + 1);
      for (int nb = 0; nb * kN < gs->C; ++nb) {
        const int nv = gs->C - nb * kN < kN ? gs->C - nb * kN : kN;
        const uint16_t* src = go + std::size_t(nb) * gb->block_stride;
        sum4_span(src + r00 * kN, src + r01 * kN, src + r10 * kN, src + r11 * kN,
                  gi + std::size_t(nb) * gs->block_stride + (std::size_t(row) + t) * kN, nv);
      }
    }
  });
}

}  // extern "C"
