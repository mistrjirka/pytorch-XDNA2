// Fused host kernels for training regions (xdna_train/regions.py).
//
// A frozen feature network with a mean-squared-error on selected taps spends
// most of its non-convolution time in memory passes: concatenating the taps,
// cloning them, subtract / square / mean, max-pool, and their backward passes.
// These kernels do the same work in one pass over each tensor.  All tensors are
// bf16 (uint16_t) in host memory; arithmetic is fp32, rounded once to bf16
// (round to nearest even).  Max-pool matches PyTorch's first-maximum rule.
//
// Each function takes the thread count explicitly so it follows the caller's
// torch.set_num_threads.

// Comparisons here never need to trap (NaN is handled explicitly); without this
// GCC will not if-convert them and the loops stay scalar.
#pragma GCC optimize("no-trapping-math")

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

inline float bits2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

inline float bf2f(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// Branch-free so that loops using it vectorize.
inline uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  const uint32_t rounded = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
  const uint32_t quiet_nan = (u >> 16) | 0x40u;
  return uint16_t(((u & 0x7fffffffu) > 0x7f800000u) ? quiet_nan : rounded);
}

}  // namespace

extern "C" {

// sum over i of (a[i] - b[i])^2, accumulated in fp32 per thread and added in double.
double xdna_region_sqdiff_sum_bf16(const uint16_t* a, const uint16_t* b, int64_t n,
                                   int threads) {
  double total = 0.0;
#pragma omp parallel num_threads(threads) reduction(+ : total)
  {
    constexpr int64_t kBlock = 4096;
    const int64_t blocks = (n + kBlock - 1) / kBlock;
#pragma omp for schedule(static)
    for (int64_t blk = 0; blk < blocks; ++blk) {
      const int64_t lo = blk * kBlock, hi = lo + kBlock < n ? lo + kBlock : n;
      float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
      for (int64_t i = lo; i < hi; ++i) {
        const float d = bf2f(a[i]) - bf2f(b[i]);
        acc += d * d;
      }
      total += double(acc);
    }
  }
  return total;
}

// r = max(a, 0) (NaN stays NaN)
void xdna_region_relu_bf16(const uint16_t* a, uint16_t* r, int64_t n, int threads) {
  constexpr int64_t kBlock = 8192;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t blk = 0; blk < blocks; ++blk) {
    const int64_t lo = blk * kBlock, hi = lo + kBlock < n ? lo + kBlock : n;
#pragma omp simd
    for (int64_t i = lo; i < hi; ++i) {
      const float v = bf2f(a[i]);
      r[i] = (v <= 0.0f) ? uint16_t(0) : a[i];
    }
  }
}

extern "C++" {
inline bool nan_or_gt(float v, float m) { return (v > m) ? true : (v != v); }
}

extern "C++" {
// Pool one row pair (r0, r1: relu'd rows of W bf16, W even) into Wo = W/2 outputs.
// Pairs of columns are read as one 32-bit word, so every access is contiguous.
__attribute__((target_clones("avx512f", "avx2", "default")))
static void pool_row_pairs(const uint16_t* r0, const uint16_t* r1, uint16_t* dst, int64_t Wo) {
  const uint32_t* w0 = reinterpret_cast<const uint32_t*>(r0);
  const uint32_t* w1 = reinterpret_cast<const uint32_t*>(r1);
#pragma omp simd
  for (int64_t x = 0; x < Wo; ++x) {
    const uint32_t a = w0[x], b = w1[x];
    const uint32_t e00 = a & 0xffffu, e01 = a >> 16, e10 = b & 0xffffu, e11 = b >> 16;
    // vertical then horizontal: same value as PyTorch's scan-order maximum
    // (NaN wins; the winning element only matters to the backward pass)
    const float f00 = bits2f(e00 << 16), f10 = bits2f(e10 << 16);
    const bool s0 = nan_or_gt(f10, f00);
    const uint32_t m0 = s0 ? e10 : e00;
    const float v0 = s0 ? f10 : f00;
    const float f01 = bits2f(e01 << 16), f11 = bits2f(e11 << 16);
    const bool s1 = nan_or_gt(f11, f01);
    const uint32_t m1 = s1 ? e11 : e01;
    const float v1 = s1 ? f11 : f01;
    dst[x] = uint16_t(nan_or_gt(v1, v0) ? m1 : m0);
  }
}

// AVX2 version: 8 output pixels (16 columns of each row) per iteration.  The
// float bit patterns are the bf16 values shifted into the high half, so a
// select on floats picks bf16 elements; same vertical-then-horizontal order and
// NaN rule as the scalar loop above.
__attribute__((target("avx2")))
static void pool_row_pairs_avx2(const uint16_t* r0, const uint16_t* r1, uint16_t* dst, int64_t Wo) {
  const uint32_t* w0 = reinterpret_cast<const uint32_t*>(r0);
  const uint32_t* w1 = reinterpret_cast<const uint32_t*>(r1);
  const __m256i hi_mask = _mm256_set1_epi32(int(0xffff0000u));
  auto pick = [](__m256 v, __m256 m) {  // v > m, or v is NaN
    return _mm256_or_ps(_mm256_cmp_ps(v, m, _CMP_GT_OQ), _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
  };
  int64_t x = 0;
  for (; x + 8 <= Wo; x += 8) {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0 + x));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1 + x));
    const __m256 f00 = _mm256_castsi256_ps(_mm256_slli_epi32(a, 16));
    const __m256 f01 = _mm256_castsi256_ps(_mm256_and_si256(a, hi_mask));
    const __m256 f10 = _mm256_castsi256_ps(_mm256_slli_epi32(b, 16));
    const __m256 f11 = _mm256_castsi256_ps(_mm256_and_si256(b, hi_mask));
    const __m256 v0 = _mm256_blendv_ps(f00, f10, pick(f10, f00));
    const __m256 v1 = _mm256_blendv_ps(f01, f11, pick(f11, f01));
    const __m256 r = _mm256_blendv_ps(v0, v1, pick(v1, v0));
    const __m256i bits = _mm256_srli_epi32(_mm256_castps_si256(r), 16);
    const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi32(bits, bits), 0x08);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), _mm256_castsi256_si128(packed));
  }
  if (x < Wo) pool_row_pairs(r0 + 2 * x, r1 + 2 * x, dst + x, Wo - x);
}
}  // extern "C++"

// r = relu(a) over [planes][H][W]; p = 2x2 stride-2 max-pool of r, [planes][H/2][W/2].
void xdna_region_relu_maxpool2_bf16(const uint16_t* a, uint16_t* r, uint16_t* p,
                                    int64_t planes, int64_t H, int64_t W, int threads) {
  const int64_t Ho = H / 2, Wo = W / 2;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t pl = 0; pl < planes; ++pl) {
    const uint16_t* ap = a + pl * H * W;
    uint16_t* rp = r + pl * H * W;
    uint16_t* pp = p + pl * Ho * Wo;
    for (int64_t y = 0; y < H; ++y) {
      const uint16_t* ar = ap + y * W;
      uint16_t* rr = rp + y * W;
#pragma omp simd
      for (int64_t x = 0; x < W; ++x) rr[x] = (bf2f(ar[x]) <= 0.0f) ? uint16_t(0) : ar[x];
    }
    for (int64_t y = 0; y < Ho; ++y) {
      const uint16_t* r0 = rp + (2 * y) * W;
      const uint16_t* r1 = r0 + W;
      if ((W & 1) == 0 && ((uintptr_t(rp) & 3) == 0)) {
        static const bool avx2 = __builtin_cpu_supports("avx2");
        if (avx2) pool_row_pairs_avx2(r0, r1, pp + y * Wo, Wo);
        else pool_row_pairs(r0, r1, pp + y * Wo, Wo);
      } else {
        for (int64_t x = 0; x < Wo; ++x) {  // scan-order maximum, NaN wins
          uint16_t m = r0[2 * x];
          float mv = bf2f(m);
          const uint16_t c[3] = {r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]};
          for (int k = 0; k < 3; ++k) {
            const float v = bf2f(c[k]);
            if (nan_or_gt(v, mv)) { m = c[k]; mv = v; }
          }
          pp[y * Wo + x] = m;
        }
      }
    }
  }
}

// out = bf16( base + scale * (a - b) );  base may be null.
void xdna_region_addtap_bf16(const uint16_t* base, const uint16_t* a, const uint16_t* b,
                             uint16_t* out, int64_t n, float scale, int threads) {
  constexpr int64_t kBlock = 8192;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t blk = 0; blk < blocks; ++blk) {
    const int64_t lo = blk * kBlock, hi = lo + kBlock < n ? lo + kBlock : n;
    if (base) {
#pragma omp simd
      for (int64_t i = lo; i < hi; ++i)
        out[i] = f2bf(bf2f(base[i]) + scale * (bf2f(a[i]) - bf2f(b[i])));
    } else {
#pragma omp simd
      for (int64_t i = lo; i < hi; ++i)
        out[i] = f2bf(scale * (bf2f(a[i]) - bf2f(b[i])));
    }
  }
}

// ReLU backward with an optional tap term:
//   out = (r <= 0) ? 0 : bf16( base + scale * (a - b) )
// base may be null (treated as 0); a and b may be null (no tap term).  The mask
// is PyTorch's threshold_backward on the relu result: a NaN result passes.
// One loop per operand combination keeps the loops branch-free (vectorizable).
extern "C++" {
template <bool HasBase, bool HasTap>
static void relu_bwd_block(const uint16_t* r, const uint16_t* base, const uint16_t* a,
                           const uint16_t* b, uint16_t* out, int64_t lo, int64_t hi,
                           float scale) {
#pragma omp simd
  for (int64_t i = lo; i < hi; ++i) {
    float g = 0.0f;
    if constexpr (HasBase) g = bf2f(base[i]);
    if constexpr (HasTap) g += scale * (bf2f(a[i]) - bf2f(b[i]));
    out[i] = (bf2f(r[i]) <= 0.0f) ? uint16_t(0) : f2bf(g);
  }
}
}  // extern "C++"

void xdna_region_relu_bwd_bf16(const uint16_t* r, const uint16_t* base, const uint16_t* a,
                               const uint16_t* b, uint16_t* out, int64_t n, float scale,
                               int threads) {
  constexpr int64_t kBlock = 8192;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t blk = 0; blk < blocks; ++blk) {
    const int64_t lo = blk * kBlock, hi = lo + kBlock < n ? lo + kBlock : n;
    if (base && a) relu_bwd_block<true, true>(r, base, a, b, out, lo, hi, scale);
    else if (base) relu_bwd_block<true, false>(r, base, a, b, out, lo, hi, scale);
    else if (a) relu_bwd_block<false, true>(r, base, a, b, out, lo, hi, scale);
    else relu_bwd_block<false, false>(r, base, a, b, out, lo, hi, scale);
  }
}

// Max-pool 2x2/2 backward: out[argmax of each window of x] = g, elsewhere 0.
// x is the pool input [planes][H][W], g the pooled gradient [planes][H/2][W/2];
// out has x's shape.  Odd trailing rows/columns get 0.  The argmax is the first
// maximum in scan order, NaN winning (PyTorch's rule).
extern "C++" {
static void pool_bwd_rows(const uint16_t* x0, const uint16_t* x1, const uint16_t* g,
                          uint16_t* o0, uint16_t* o1, int64_t Wo) {
  const uint32_t* w0 = reinterpret_cast<const uint32_t*>(x0);
  const uint32_t* w1 = reinterpret_cast<const uint32_t*>(x1);
  uint32_t* q0 = reinterpret_cast<uint32_t*>(o0);
  uint32_t* q1 = reinterpret_cast<uint32_t*>(o1);
#pragma omp simd
  for (int64_t x = 0; x < Wo; ++x) {
    const uint32_t a = w0[x], b = w1[x];
    const float e0 = bf2f(uint16_t(a)), e1 = bf2f(uint16_t(a >> 16));
    const float e2 = bf2f(uint16_t(b)), e3 = bf2f(uint16_t(b >> 16));
    float mv = e0;
    int best = 0;
    const bool c1 = nan_or_gt(e1, mv);
    best = c1 ? 1 : best;
    mv = c1 ? e1 : mv;
    const bool c2 = nan_or_gt(e2, mv);
    best = c2 ? 2 : best;
    mv = c2 ? e2 : mv;
    best = nan_or_gt(e3, mv) ? 3 : best;
    const uint32_t gv = g[x];
    q0[x] = (best == 0 ? gv : 0u) | (best == 1 ? gv << 16 : 0u);
    q1[x] = (best == 2 ? gv : 0u) | (best == 3 ? gv << 16 : 0u);
  }
}
}  // extern "C++"

void xdna_region_maxpool2_bwd_bf16(const uint16_t* x, const uint16_t* g, uint16_t* out,
                                   int64_t planes, int64_t H, int64_t W, int threads) {
  const int64_t Ho = H / 2, Wo = W / 2;
  const bool fast = (W & 1) == 0 && (uintptr_t(x) & 3) == 0 && (uintptr_t(out) & 3) == 0;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t pl = 0; pl < planes; ++pl) {
    const uint16_t* xp = x + pl * H * W;
    const uint16_t* gp = g + pl * Ho * Wo;
    uint16_t* op = out + pl * H * W;
    if (fast) {
      for (int64_t y = 0; y < Ho; ++y)
        pool_bwd_rows(xp + 2 * y * W, xp + (2 * y + 1) * W, gp + y * Wo, op + 2 * y * W,
                      op + (2 * y + 1) * W, Wo);
      if (H & 1) std::memset(op + (H - 1) * W, 0, size_t(W) * sizeof(uint16_t));
      continue;
    }
    std::memset(op, 0, size_t(H * W) * sizeof(uint16_t));
    for (int64_t y = 0; y < Ho; ++y) {
      for (int64_t xo = 0; xo < Wo; ++xo) {
        const int64_t base = (2 * y) * W + 2 * xo;
        const int64_t off[4] = {0, 1, W, W + 1};
        int best = 0;
        float mv = bf2f(xp[base]);
        for (int k = 1; k < 4; ++k) {
          const float v = bf2f(xp[base + off[k]]);
          if (nan_or_gt(v, mv)) { best = k; mv = v; }
        }
        op[base + off[best]] = gp[y * Wo + xo];
      }
    }
  }
}

}  // extern "C"
