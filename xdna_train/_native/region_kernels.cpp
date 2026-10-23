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

// ---------------------------------------------------------------------------
// Training BatchNorm2d + LeakyReLU on contiguous NCHW bf16.
//
// Statistics: each (n, c) plane is reduced in fp32 (mean, then sum of squared
// deviations; the plane is cache resident, so the second pass is cheap) and the
// N planes of a channel are merged in double with Chan's formula, so the result
// does not depend on a sum/sumsq cancellation.  Padding is not an issue here: the
// tensor is plain NCHW, every element is valid.  The fused forward rounds the
// activation once (after BN and LeakyReLU); eager rounds the BN output to bf16
// first.  Backward uses y > 0 as the LeakyReLU mask (slope > 0, so y > 0 iff the
// pre-activation is > 0; a pre-activation of exactly 0 takes the slope branch
// like PyTorch's leaky_relu_backward).  For ReLU (relu != 0) a NaN activation
// passes its gradient, as PyTorch's relu backward does.
// ---------------------------------------------------------------------------

extern "C++" {
inline void plane_mean_m2(const uint16_t* x, int64_t hw, float& mean, float& m2) {
  float s = 0.0f;
#pragma omp simd reduction(+ : s)
  for (int64_t i = 0; i < hw; ++i) s += bf2f(x[i]);
  const float m = s / float(hw);
  float q = 0.0f;
#pragma omp simd reduction(+ : q)
  for (int64_t i = 0; i < hw; ++i) {
    const float d = bf2f(x[i]) - m;
    q += d * d;
  }
  mean = m;
  m2 = q;
}
}  // extern "C++"

extern "C" {

// Training statistics only.  If update != 0 the running statistics (bf16)
// move by `factor` towards the batch mean and the unbiased batch variance.
// mean_out / invstd_out are fp32 [C]; var_out (may be null) gets the biased
// variance.
void xdna_region_bn_stats_bf16(const uint16_t* x, uint16_t* rm, uint16_t* rv, float* mean_out,
                               float* invstd_out, float* var_out, int64_t N, int64_t C,
                               int64_t HW, float eps, float factor, int update, int threads) {
  float* pm = new float[size_t(N * C) * 2];
  float* pq = pm + N * C;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t pl = 0; pl < N * C; ++pl) plane_mean_m2(x + pl * HW, HW, pm[pl], pq[pl]);
  const double M = double(N) * double(HW);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t c = 0; c < C; ++c) {
    double n_acc = 0.0, mean = 0.0, m2 = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      const double nb = double(HW), pmean = double(pm[n * C + c]);
      const double tot = n_acc + nb, delta = pmean - mean;
      mean += delta * nb / tot;
      m2 += double(pq[n * C + c]) + delta * delta * n_acc * nb / tot;
      n_acc = tot;
    }
    const double var = m2 / M;
    mean_out[c] = float(mean);
    if (var_out) var_out[c] = float(var);
    invstd_out[c] = float(1.0 / std::sqrt(var + double(eps)));
    if (update) {
      const double unbiased = M > 1.0 ? var * M / (M - 1.0) : var;
      rm[c] = f2bf(float((1.0 - double(factor)) * double(bf2f(rm[c])) + double(factor) * mean));
      rv[c] = f2bf(float((1.0 - double(factor)) * double(bf2f(rv[c])) + double(factor) * unbiased));
    }
  }
  delete[] pm;
}

// Forward.  w and b may be null (affine=False).  Statistics and running update
// as xdna_region_bn_stats_bf16.
void xdna_region_bn_fwd_bf16(const uint16_t* x, const uint16_t* w, const uint16_t* b,
                             uint16_t* rm, uint16_t* rv, uint16_t* y, float* mean_out,
                             float* invstd_out, int64_t N, int64_t C, int64_t HW, float eps,
                             float slope, float factor, int update, int threads) {
  xdna_region_bn_stats_bf16(x, rm, rv, mean_out, invstd_out, nullptr, N, C, HW, eps, factor,
                            update, threads);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t pl = 0; pl < N * C; ++pl) {
    const int64_t c = pl % C;
    const float scale = (w ? bf2f(w[c]) : 1.0f) * invstd_out[c];
    const float shift = (b ? bf2f(b[c]) : 0.0f) - mean_out[c] * scale;
    const uint16_t* xp = x + pl * HW;
    uint16_t* yp = y + pl * HW;
#pragma omp simd
    for (int64_t i = 0; i < HW; ++i) {
      const float z = bf2f(xp[i]) * scale + shift;
      yp[i] = f2bf(z > 0.0f ? z : z * slope);
    }
  }
}

// Backward of BN + LeakyReLU.  dw_out / db_out (bf16 [C]) get sum(dz * xhat) and
// sum(dz); dx (bf16, may be null when not needed) gets the input gradient.
void xdna_region_bn_bwd_bf16(const uint16_t* dy, const uint16_t* y, const uint16_t* x,
                             const uint16_t* w, const float* mean, const float* invstd,
                             uint16_t* dx, uint16_t* dw_out, uint16_t* db_out, int64_t N,
                             int64_t C, int64_t HW, float slope, int relu, int threads) {
  float* ps = new float[size_t(N * C) * 2];
  float* pt = ps + N * C;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t pl = 0; pl < N * C; ++pl) {
    const int64_t c = pl % C;
    const float mc = mean[c], is = invstd[c];
    const uint16_t* dyp = dy + pl * HW;
    const uint16_t* yp = y + pl * HW;
    const uint16_t* xp = x + pl * HW;
    float s1 = 0.0f, s2 = 0.0f;
#pragma omp simd reduction(+ : s1, s2)
    for (int64_t i = 0; i < HW; ++i) {
      const float g = bf2f(dyp[i]);
      const float yv = bf2f(yp[i]);
      // ReLU passes the gradient for NaN (threshold_backward: !(y <= 0)); LeakyReLU
      // sends NaN to the slope branch (y > 0).  Equal for every other value.
      const float dz = ((yv > 0.0f) | (relu & (yv != yv))) ? g : g * slope;
      s1 += dz;
      s2 += dz * ((bf2f(xp[i]) - mc) * is);
    }
    ps[pl] = s1;
    pt[pl] = s2;
  }
  const double M = double(N) * double(HW);
  float* k = new float[size_t(C) * 2];
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t c = 0; c < C; ++c) {
    double a = 0.0, bsum = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      a += double(ps[n * C + c]);
      bsum += double(pt[n * C + c]);
    }
    if (db_out) db_out[c] = f2bf(float(a));
    if (dw_out) dw_out[c] = f2bf(float(bsum));
    k[c] = float(a / M);
    k[C + c] = float(bsum / M);
  }
  delete[] ps;
  if (dx) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t pl = 0; pl < N * C; ++pl) {
      const int64_t c = pl % C;
      const float mc = mean[c], is = invstd[c];
      const float ws = (w ? bf2f(w[c]) : 1.0f) * is;
      const float k1 = k[c], k2 = k[C + c];
      const uint16_t* dyp = dy + pl * HW;
      const uint16_t* yp = y + pl * HW;
      const uint16_t* xp = x + pl * HW;
      uint16_t* dxp = dx + pl * HW;
#pragma omp simd
      for (int64_t i = 0; i < HW; ++i) {
        const float g = bf2f(dyp[i]);
        const float yv = bf2f(yp[i]);
      // ReLU passes the gradient for NaN (threshold_backward: !(y <= 0)); LeakyReLU
      // sends NaN to the slope branch (y > 0).  Equal for every other value.
      const float dz = ((yv > 0.0f) | (relu & (yv != yv))) ? g : g * slope;
        const float xhat = (bf2f(xp[i]) - mc) * is;
        dxp[i] = f2bf(ws * (dz - k1 - xhat * k2));
      }
    }
  }
  delete[] k;
}

}  // extern "C"
