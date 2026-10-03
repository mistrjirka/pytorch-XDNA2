#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

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

#pragma omp parallel for collapse(3) schedule(static)
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

#pragma omp parallel for collapse(3) schedule(static)
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

#pragma omp parallel for collapse(2) schedule(static)
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

#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp parallel for collapse(2) schedule(static)
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

#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp parallel for collapse(3) schedule(static)
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
#pragma omp parallel for collapse(3) schedule(static)
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

#pragma omp parallel for schedule(static)
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
#pragma omp parallel for collapse(2) schedule(static)
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

#pragma omp parallel for schedule(static)
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

#pragma omp parallel for schedule(static)
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
#pragma omp parallel for collapse(2) schedule(static)
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
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(n);
#else
  (void)n;
#endif
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
#pragma omp parallel for collapse(3) schedule(static)
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
