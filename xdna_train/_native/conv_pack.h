#pragma once
#include <cstdint>

// One channel range of a lazy value, as the conv3x3w pack and the materialize
// kernel read it (built from XdnaLazySegment in torch_xdna_device.cpp):
// channels [c0, c0 + channels) of the value are
// act(up(src) * scale + shift), with up() a nearest upsample by an integer
// factor.  A plain tensor is one segment with up 1 and no pointwise.
struct XdnaPackSegment {
  const uint16_t* src;  // NCHW bf16 [B][channels][H / up][W / up], contiguous
  int c0, channels;
  int up;
  const float* scale;   // per segment channel, nullptr: no affine
  const float* shift;
  int act;              // 0 none, 1 LeakyReLU with slope, 2 ReLU (XdnaAct)
  float slope;
  // Stage 4: when rows is set, the source is still the conv3x3w output rows
  // the NPU wrote (src unused): [blocks][grid rows][128] bf16 in the strip
  // layout of a (H / up) x (W / up) image, channel k in block k / 128.
  const uint16_t* rows;
  long block_stride;    // elements between 128-channel blocks
  int rows_s, rows_so, rows_wp;
};

// The grid of a tensor kept in conv3x3w output rows (stage 4): B x C x H x W
// in the strip layout of conv3x3w_strips(W) (S strips of So columns, padded
// row width wp), channel c at block c / 128 of block_stride elements.
struct XdnaRowsGeom {
  int B, C, H, W, S, So, wp;
  long block_stride;
};

extern "C" {
// Groups [g0, g0 + ng) of the packed conv3x3w input (strip layout) of the
// value the segments describe; see xdna_conv3x3w_pack_input_strips_range_bf16.
void xdna_conv3x3w_pack_segments_bf16(
    const XdnaPackSegment* segs, int nsegs, uint16_t* dst, int B, int C, int H, int W, int S,
    int So, int halo_zero, int chunks, int g0, int ng);
// Unpack of conv3x3w output rows [p0, p1) (strip layout) into channels
// [c_start, c_start + cout_valid) of NCHW dst; with sum / sumsq set, also adds
// the per-channel sum and sum of squares of those outputs.  dst may be null.
void xdna_conv3x3w_unpack_output_strips_range_stats_bf16(
    const uint16_t* c, uint16_t* dst, int B, int H, int W, int S, int So, int cout_valid,
    int C_total, int c_start, int p0, int p1, double* sum, double* sumsq);
// Elementwise and BatchNorm backward on tensors kept in rows of the same grid
// (stage 4).  Only valid rows are read or written; padding rows stay as they
// are.  A pointwise (scale/shift: per channel, nullptr = none) applies to x.
// mode 1: LeakyReLU backward, dx = dy * (z > 0 ? 1 : slope), z = affine(x).
// mode 2: ReLU backward (threshold_backward on relu(z)), dx = z <= 0 ? 0 : dy.
void xdna_rows_act_backward_bf16(
    const XdnaRowsGeom* g, const uint16_t* dy, const uint16_t* x, uint16_t* dx,
    const float* scale, const float* shift, int mode, float slope);
// ATen's training BatchNorm backward: per channel sum(dy) and
// sum((x - mean) * dy) (double here), dx = (dy - sum/N - (x - mean) * k) *
// invstd * w with k = dotp * invstd^2 / N.  w may be null (1); dx may be null.
void xdna_rows_bn_backward_bf16(
    const XdnaRowsGeom* g, const uint16_t* dy, const uint16_t* x, uint16_t* dx,
    const float* mean, const float* invstd, const float* w, float* dgamma, float* dbeta);
// out = pw_a(a) + pw_b(b), each rounded to bf16 first (the lazy add).
void xdna_rows_add_bf16(
    const XdnaRowsGeom* g, const uint16_t* a, const float* a_scale, const float* a_shift,
    int a_act, float a_slope, const uint16_t* b, const float* b_scale, const float* b_shift,
    int b_act, float b_slope, uint16_t* out);
// gi (rows on grid gs, B x C x H x W) = upsample_nearest2d_backward of go (rows
// on grid gb, B x C x 2H x 2W): each 2x2 block summed in fp32 in raster order,
// as the eager kernel does.
void xdna_rows_upsample2x_backward_bf16(
    const XdnaRowsGeom* gb, const uint16_t* go, const XdnaRowsGeom* gs, uint16_t* gi);
// dst (contiguous NCHW [B][C][H][W] bf16) = the value the segments describe.
void xdna_lazy_segments_write_bf16(
    const XdnaPackSegment* segs, int nsegs, uint16_t* dst, int B, int C, int H, int W);
}
