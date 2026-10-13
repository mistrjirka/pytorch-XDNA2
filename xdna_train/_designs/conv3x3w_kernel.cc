// XDNA2 conv3x3 window kernel: bfp16ebs8 activations x bfp16ebs8 weights -> bf16.
// The host converts the activations to bfp16 once (conv3x3w_host.pack_input_bfp),
// so the inner loop is block loads and mac_8x8_8x8T only.  Converting bf16
// activations in the kernel instead cost each 8x8 block 8 conversions per tap
// and took 3.1 ms for a pass that now takes 1.7 ms.
// MAC microkernel (2x2 output blocking) derived from AMD mlir-aie
// aie_kernels/aie2p/mm_bfp_mixed.cc.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8);
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) aie::store_v(c, zeros);
}

// Window: [8 channel blocks][WINB rows][9 bytes].  The A block of row block z
// and channel block i for a tap is rows tap_rows + 8z .. +7 of block i: 72
// contiguous bytes at any byte offset.  Rows z and z+1 are consecutive, so one
// load stream pops both.
//
// Hardware rule (found on the device, not documented): the load stream fills
// 64-byte lines and holds two of them, so a pair of 72-byte pops is exact only
// when it starts at most 48 bytes into its line (a later start loses the last
// block).  The caller keeps a second copy of the window (prep_window) at
// pShift, which is congruent to pWin + 16 modulo 64; a pair that starts late in
// its line is read from the copy, where it starts within the first 15 bytes.
//
// Where prep_window puts the copy, as an offset into the scratch buffer: from
// the two buffers' own addresses, so it holds for any placement.  Needs both
// buffers 16-byte aligned and 80 bytes of slack in the scratch.
inline unsigned shift_base(const uint8_t *pWin, const uint8_t *pS) {
  return 16 + (unsigned)(((uintptr_t)pWin - (uintptr_t)pS) & 63);
}

template <unsigned rowA, unsigned colA, unsigned colB, unsigned winb>
void matmul_window_bfp(const uint8_t *__restrict pWin, const uint8_t *__restrict pShift,
                       const bfp16ebs8 *__restrict pB, bfloat16 *__restrict pC,
                       int tap_rows) {
  constexpr unsigned sizeC = 64;
  using BS = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;
  for (unsigned z = 0; z < rowA; z += 2) {
    bfloat16 *__restrict pC1 = pC + (z * colB) * sizeC;
    bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * sizeC;
    // Start of the row pair in each channel block, from whichever copy
    // places it early enough in its line (independent of j).
    const uint8_t *pA[colA];
#pragma unroll
    for (unsigned i = 0; i < colA; ++i) {
      const unsigned off = (tap_rows + z * 8 + i * winb) * 9;
      pA[i] = (((uintptr_t)(pWin + off)) & 63) > 48 ? pShift + off : pWin + off;
    }
    AIE_PREPARE_FOR_PIPELINING
    for (unsigned j = 0; j < colB; j += 2) {
      BS bs(pB);
      bs.seek(j * colA);
      aie::accum<accfloat, sizeC> c00(aie::load_v<sizeC>(pC1));
      aie::accum<accfloat, sizeC> c01(aie::load_v<sizeC>(pC1 + sizeC));
      aie::accum<accfloat, sizeC> c10(aie::load_v<sizeC>(pC2));
      aie::accum<accfloat, sizeC> c11(aie::load_v<sizeC>(pC2 + sizeC));
#pragma unroll
      for (unsigned i = 0; i < colA; ++i) {
        BS as((const bfp16ebs8 *)pA[i]);
        const auto A0 = as.pop();
        const auto A1 = as.pop();
        const auto B0 = bs.pop();
        const auto B1 = bs.pop();
        c00 = mac_8x8_8x8T(A0, B0, c00);
        c01 = mac_8x8_8x8T(A0, B1, c01);
        c10 = mac_8x8_8x8T(A1, B0, c10);
        c11 = mac_8x8_8x8T(A1, B1, c11);
      }
      aie::store_v(pC1, c00.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC1, c01.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC2, c10.template to_vector<bfloat16>()); pC2 += sizeC;
      aie::store_v(pC2, c11.template to_vector<bfloat16>()); pC2 += sizeC;
    }
  }
}

extern "C" {

#ifdef WINDOW_ONLY
#ifndef WINB
#error "WINB (window rows) must be defined"
#endif
// C (+)= the tap at row offset tap_rows; pS is the shifted copy of the window.
void matmul_window(const uint8_t *__restrict pWin, const uint8_t *__restrict pS,
                   const bfp16ebs8 *__restrict pB, bfloat16 *__restrict pC,
                   int32_t tap_rows) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  static_assert(DIM_M % 16 == 0 && DIM_K == 64 && DIM_N % 16 == 0);
  matmul_window_bfp<DIM_M / 8, DIM_K / 8, DIM_N / 8, WINB>(
      pWin, pS + shift_base(pWin, pS), pB, pC, tap_rows);
  aie::set_rounding(saved_rounding);
}
#endif

#ifdef PREP_ONLY
// The shifted copy of the window: pS + shift_base(pWin, pS) (16-byte vectors
// keep both sides aligned, since the offset is a multiple of 16).
void prep_window(const uint8_t *__restrict pWin, uint8_t *__restrict pS) {
  constexpr unsigned n = WINB * 72 / 16;
  uint8_t *__restrict d = pS + shift_base(pWin, pS);
  AIE_PREPARE_FOR_PIPELINING
  for (unsigned k = 0; k < n; ++k)
    aie::store_v(d + 16 * k, aie::load_v<16>(pWin + 16 * k));
}
#endif

#ifdef ZERO_ONLY
void zero_kernel_bf16(bfloat16 *__restrict c) {
  zero_vectorized<bfloat16, DIM_M, DIM_N>(c);
}
#endif
}
