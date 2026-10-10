// XDNA2 conv3x3 weight-gradient window kernel: bf16 x bf16 -> fp32.
//
// dW[dy][dx][ci][co] = sum_p X[p + dy*(W+2) + dx][ci] * dY[p][co] over
// padded-grid pixels p.  A core owns one kernel row dy, 64 input channels
// and 32 output channels, and keeps that tile's fp32 sums in L1 across the
// whole dispatch (a bf16 running sum over ~50k pixels would lose the
// gradient).  Per call it adds one 64-pixel step:
//   win: X rows p0+dy*(W+2) .. +66, 8-channel blocked [8][66][8]; tap dx
//        is a 1-row shift, read with unaligned loads (aligned loads drop the
//        low address bits).
//   dy:  dY rows p0 .. +64, 8-channel blocked [4][64][8]: each 8x8 block is
//        directly the mmul B operand [8 p][8 co].
//   acc: [3 dx][8 ci blk][4 co blk][8 ci][8 co] fp32.
// The reduction runs over pixels, so the X block [8 p][8 ci] is transposed
// to the mmul A operand [8 ci][8 p]; each transpose feeds 4 MACs.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef WIN_R
#define WIN_R 66
#endif

extern "C" {

#ifdef ZERO_ONLY
void zero_dw_f32(float *__restrict acc) {
  const aie::vector<float, 32> zeros = aie::zeros<float, 32>();
  for (unsigned i = 0; i < 3 * 8 * 4 * 64; i += 32) aie::store_v(acc + i, zeros);
}
#endif

#ifdef DW_ONLY
// AIE2P has no native exact-bf16 matmul (aie_api emulates it with element-wise
// MACs, ~16 MAC/cycle); bfp16ebs8 blocks run at 512.  Both operands are
// converted with their 8-element blocks along the pixel (reduction) axis:
// A = X^T [8 ci][8 p], B = dY^T [8 co][8 p] (mac_8x8_8x8T takes B as [n][k]).
// dY blocks are converted once per call into `dyb` and reused by all 24
// (dx, ci block) tiles; each X block is converted once and feeds 4 MACs.
void dw_window(const bfloat16 *__restrict win, const bfloat16 *__restrict dy,
               float *__restrict acc, bfp16ebs8 *__restrict dyb) {
#ifdef DW_NOP
  return;
#endif
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(dyb);
    for (unsigned z = 0; z < 8; ++z)
      for (unsigned j = 0; j < 4; ++j) {
        aie::accum<accfloat, 64> a;
        a = aie::transpose(aie::load_v<64>(dy + j * 512 + z * 64), 8, 8);
        out << a;
      }
  }
  for (unsigned dx = 0; dx < 3; ++dx) {
    AIE_PREPARE_FOR_PIPELINING
    for (unsigned i = 0; i < 8; ++i) {
      const bfloat16 *__restrict pa = win + i * WIN_R * 8 + dx * 8;
      float *__restrict pc = acc + (dx * 8 + i) * 4 * 64;
      aie::accum<accfloat, 64> c0(aie::load_v<64>(pc));
      aie::accum<accfloat, 64> c1(aie::load_v<64>(pc + 64));
      aie::accum<accfloat, 64> c2(aie::load_v<64>(pc + 128));
      aie::accum<accfloat, 64> c3(aie::load_v<64>(pc + 192));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs(dyb);
      for (unsigned z = 0; z < 8; ++z) {
        aie::accum<accfloat, 64> at;
        at = aie::transpose(aie::load_unaligned_v<64>(pa + z * 64, 8), 8, 8);
        const aie::block_vector<bfp16ebs8, 64> a(::to_v64bfp16ebs8(at));
        c0 = mac_8x8_8x8T(a, bs.pop(), c0);
        c1 = mac_8x8_8x8T(a, bs.pop(), c1);
        c2 = mac_8x8_8x8T(a, bs.pop(), c2);
        c3 = mac_8x8_8x8T(a, bs.pop(), c3);
      }
      aie::store_v(pc, c0.template to_vector<float>());
      aie::store_v(pc + 64, c1.template to_vector<float>());
      aie::store_v(pc + 128, c2.template to_vector<float>());
      aie::store_v(pc + 192, c3.template to_vector<float>());
    }
  }
  aie::set_rounding(saved_rounding);
}
#endif
}
