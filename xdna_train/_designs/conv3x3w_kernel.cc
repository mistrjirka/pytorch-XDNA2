// XDNA2 conv3x3 window kernel: bf16 activations x bfp16ebs8 weights -> bf16.
// Microkernel derived from AMD mlir-aie's Peano ATB config1 k4/RNE kernel
// (2x2 output blocking, 4-stage K pump, round-to-nearest-even).
// Keeps config1's 4-way asymmetric C accumulation, but uses the generic
// Peano-compatible 2x2 mixed microkernel instead of Chess register-placement
// annotations. Derived from AMD mlir-aie aie_kernels/aie2p/mm_bfp_mixed.cc.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8);
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) aie::store_v(c, zeros);
}

// Window variant: A is read straight out of a per-core input window held in
// L1 in 8-channel-blocked layout [64/8][WIN_R][8] (one 64-channel k-step).
// Output row p of the core's tile reads window row p + tap_rows, so every 3x3
// tap is the same window at a different row offset -- no DMA expansion.
// A tap offset moves the pointer by 16-byte rows; aligned vector loads drop the
// low address bits (silently reading the wrong rows), so A uses unaligned loads.
// 8x8 block (row block z, channel block i) starts at
//   win + tap_rows*8 + z*64 + i*WIN_R*8.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t, unsigned win_r>
void matmul_2x2_window(const bfloat16 *__restrict pWin,
                       const bfp16ebs8 *__restrict pB,
                       bfloat16 *__restrict pC) {
  constexpr unsigned sizeC = r * t;
  constexpr unsigned blk_stride = win_r * 8;

  AIE_PREPARE_FOR_PIPELINING
  for (unsigned z = 0; z < rowA; z += 2) {
    bfloat16 *__restrict pC1 = pC + (z * colB) * sizeC;
    bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * sizeC;

    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA1 = pWin + z * 64;
      const bfloat16 *__restrict pA2 = pWin + (z + 1) * 64;

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pBstream(pB);
      pBstream.seek(j * colA);

      aie::accum<accfloat, sizeC> c00(aie::load_v<sizeC>(pC1));
      aie::accum<accfloat, sizeC> c01(aie::load_v<sizeC>(pC1 + sizeC));
      aie::accum<accfloat, sizeC> c10(aie::load_v<sizeC>(pC2));
      aie::accum<accfloat, sizeC> c11(aie::load_v<sizeC>(pC2 + sizeC));

      static_assert(colA % 4 == 0);
      for (unsigned i = 0; i < colA; i += 4) {
        const auto A00 = aie::load_unaligned_v<64>(pA1, 8); pA1 += blk_stride;
        const auto A10 = aie::load_unaligned_v<64>(pA2, 8); pA2 += blk_stride;
        aie::accum<accfloat, 64> a00; a00 = A00;
        aie::accum<accfloat, 64> a10; a10 = A10;
        const auto B00 = pBstream.pop();
        const auto B10 = pBstream.pop();
        const aie::block_vector<bfp16ebs8, 64> A0b0(::to_v64bfp16ebs8(a00));
        const aie::block_vector<bfp16ebs8, 64> A1b0(::to_v64bfp16ebs8(a10));
        const auto A01 = aie::load_unaligned_v<64>(pA1, 8); pA1 += blk_stride;
        const auto A11 = aie::load_unaligned_v<64>(pA2, 8); pA2 += blk_stride;
        aie::accum<accfloat, 64> a01; a01 = A01;
        aie::accum<accfloat, 64> a11; a11 = A11;
        const auto B01 = pBstream.pop();
        const auto B11 = pBstream.pop();
        const aie::block_vector<bfp16ebs8, 64> A0b1(::to_v64bfp16ebs8(a01));
        const aie::block_vector<bfp16ebs8, 64> A1b1(::to_v64bfp16ebs8(a11));
        const auto A02 = aie::load_unaligned_v<64>(pA1, 8); pA1 += blk_stride;
        const auto A12 = aie::load_unaligned_v<64>(pA2, 8); pA2 += blk_stride;
        aie::accum<accfloat, 64> a02; a02 = A02;
        aie::accum<accfloat, 64> a12; a12 = A12;
        const auto B02 = pBstream.pop();
        const auto B12 = pBstream.pop();
        const aie::block_vector<bfp16ebs8, 64> A0b2(::to_v64bfp16ebs8(a02));
        const aie::block_vector<bfp16ebs8, 64> A1b2(::to_v64bfp16ebs8(a12));
        const auto A03 = aie::load_unaligned_v<64>(pA1, 8); pA1 += blk_stride;
        const auto A13 = aie::load_unaligned_v<64>(pA2, 8); pA2 += blk_stride;
        aie::accum<accfloat, 64> a03; a03 = A03;
        aie::accum<accfloat, 64> a13; a13 = A13;
        const auto B03 = pBstream.pop();
        const auto B13 = pBstream.pop();
        const aie::block_vector<bfp16ebs8, 64> A0b3(::to_v64bfp16ebs8(a03));
        const aie::block_vector<bfp16ebs8, 64> A1b3(::to_v64bfp16ebs8(a13));
        c00 = mac_8x8_8x8T(A0b0, B00, c00);
        c01 = mac_8x8_8x8T(A0b0, B10, c01);
        c10 = mac_8x8_8x8T(A1b0, B00, c10);
        c11 = mac_8x8_8x8T(A1b0, B10, c11);
        c00 = mac_8x8_8x8T(A0b1, B01, c00);
        c01 = mac_8x8_8x8T(A0b1, B11, c01);
        c10 = mac_8x8_8x8T(A1b1, B01, c10);
        c11 = mac_8x8_8x8T(A1b1, B11, c11);
        c00 = mac_8x8_8x8T(A0b2, B02, c00);
        c01 = mac_8x8_8x8T(A0b2, B12, c01);
        c10 = mac_8x8_8x8T(A1b2, B02, c10);
        c11 = mac_8x8_8x8T(A1b2, B12, c11);
        c00 = mac_8x8_8x8T(A0b3, B03, c00);
        c01 = mac_8x8_8x8T(A0b3, B13, c01);
        c10 = mac_8x8_8x8T(A1b3, B03, c10);
        c11 = mac_8x8_8x8T(A1b3, B13, c11);
      }

      aie::store_v(pC1, c00.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC1, c01.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC2, c10.template to_vector<bfloat16>()); pC2 += sizeC;
      aie::store_v(pC2, c11.template to_vector<bfloat16>()); pC2 += sizeC;
    }
  }
}

extern "C" {
#ifndef DIM_M
#define DIM_M 128
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 128
#endif


#ifdef WINDOW_ONLY
#ifndef WIN_R
#error "WIN_R (window rows) must be defined"
#endif
void matmul_window(const bfloat16 *__restrict pWin,
                   const bfp16ebs8 *__restrict pB,
                   bfloat16 *__restrict pC, int32_t tap_rows) {
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned r = 8, s = 8, t = 8;
  static_assert(DIM_M % 16 == 0 && DIM_K == 64 && DIM_N % 16 == 0);
  matmul_2x2_window<DIM_M / r, DIM_K / s, DIM_N / t, r, s, t, WIN_R>(
      pWin + tap_rows * 8, pB, pC);
  aie::set_rounding(saved_rounding);
}
#endif

#ifdef ZERO_ONLY
void zero_kernel_bf16(bfloat16 *__restrict c) {
  zero_vectorized<bfloat16, DIM_M, DIM_N>(c);
}
#endif
}
