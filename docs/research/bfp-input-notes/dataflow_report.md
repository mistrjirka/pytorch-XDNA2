# NPU conv dataflow: decode/re-encode chains, overlap, unpack cost

Paths: D = /home/jirka/programy/torch-xdna2-bfp/xdna_train/_native/torch_xdna_device.cpp, P = .../conv_pack.cpp.
Model: /home/jirka/programovani/magistr/nis/lab02_cnn/models.py:65-106 (MyNet).
Numbers = per-call means over the 16 calls (6 warm + 10) in /tmp/kernel_analysis/tr_bfp.log (tr_old in brackets).

## 0. Which convs are on the NPU
Only 3 of MyNet's convs take the conv3x3w path (stride-1, 36/18 wide); conv0/1/2 (stride 2 or cin 3) and the 9x9 ResNet convs stay on CPU.
Per step (1 call each, so the per-call means are per-step costs):

| call | shape (log key) | total | wait | pack | unpack | sync |
|---|---|---|---|---|---|---|
| fwd upconv2.conv | 18x18 cin256 cout128 g8 | 4.48 | 3.04 | 0.55 | 0.24 | 0.04 |
| fwd upconv1.conv | 36x36 cin256 cout128 g29 | 5.23 | 0.12 | 2.76 | 0.90 | 0.15 |
| fwd conv11.conv | 36x36 cin195 cout64 g29 | 3.97 | 0.28 | 2.36 | 0.53 | 0.15 |
| dX conv11 | 36x36 cin64 cout195 | 4.24 | 0.38 | 0.70 | 1.93 | 0.33 |
| dX upconv1 | 36x36 cin128 cout256 | 5.14 | 0.25 | 1.34 | 2.31 | 0.30 |
| dX upconv2 | 18x18 cin128 cout256 | 1.73 | 0.38 | 0.26 | 0.45 | 0.09 |
| sum | | 24.8 | 4.45 | 7.97 | 6.36 | 1.06 |

pack+unpack = 14.3 ms/step, wait = 4.4 ms. About 4.5 ms/step of total is in none of the traced buckets
(total - wait - pack - unpack - sync: 0.6, 1.3, 0.65, 0.9, 0.95, 0.5). That is out-alloc, ensure_host_current,
weight-slot lookup, shim start, bias add. It is worth a separate look (D:2043-2060, 2113-2190).

wait is NOT NPU compute. 3.04 of the 4.45 ms is the first call, fwd 18x18, whose NPU "wait" is 3.0 ms in every
step although it has 4x fewer groups than the 36x36 call (wait 0.12). The comment at D:~2330 says an
xclbin hardware-context switch costs ~2.6 ms, and the dW xclbin (Conv3x3wDw) ran just before. This is my
inference (no counter shows it), medium confidence. The NPU is otherwise idle: wait is 0.1-0.4 ms
in the other five calls, so the step is host-bound, not NPU-bound.

## 1. Decode -> re-encode chains
Every NPU conv boundary goes through NCHW bf16 in host memory. run() (D:2030-2190) always does
`cpu_alias(x)` -> pack_input_strips_range_bfp16 and always allocates an NCHW `out` and unpacks into it (D:2058-2064, 2172).
There is no NPU-layout-to-NPU-layout path. All three fwd->fwd links go through BN + LeakyReLU + cat (+ upsample):

Forward (models.py:87-106; BN/LReLU/cat/upsample are CPU handlers at D:3084, 3702, 3839, 4637-4639):
- upconv2.conv input = upsample(cat[x_res, x2]) (CPU fast-cat/upsample, D:3464+) -> pack 0.55.
  Output unpack 0.24 -> BN(train stats) -> LReLU -> cat[., x1] -> upsample 2x -> pack for upconv1.
- upconv1.conv: pack 2.76 (cin256) <- upsample(cat); unpack 0.90 -> BN -> LReLU -> cat[., x0, x] -> pack for conv11.
- conv11.conv: pack 2.36 (cin195); unpack 0.53 -> BN -> LReLU -> conv12 (1x1, not on this path).
Redundant pairs (unpack of conv k + pack of conv k+1): upconv2->upconv1: 0.24+2.76 = 3.0 ms;
upconv1->conv11: 0.90+2.36 = 3.3 ms. Neither is a pure re-encode: BN needs batch statistics (a full reduction), and
autograd needs the NCHW BN input/output for BN and LReLU backward. So the pair cannot be dropped
without a fused conv-BN-LReLU op. Elementwise-only chains (bias, LReLU alone) do not exist here (convs have bias=False).

Forward input is NOT packed twice. With keep_input (D:2205-2236) the one forward pack also writes the bf16 dW-layout copy `x16`
(P:~830-870, dst16), attached to the input's allocation (attach_packed, D:1697) and taken in backward
(take_packed, D:4278, tag = pointer+version+write_epoch). The cost is the extra x16 writes inside the 2.76/2.36 ms
fwd packs (26 MB of bf16 at cin256). It is not a separate repack. A tag mismatch (in-place write) would repack in
pack_operands (D:2389-2408, phase "npu_conv3x3w.dw.pack"). That phase never appears in the logs, so I assume reuse works.

Backward:
- dY is NOT packed twice. The dX run packs dY once and keeps it as `d_buf` (keep_dy, D:4364-4368), which is exactly dW's layout
  (same chunks), when strips==1 (all W<=36 here, D:2392-2396). dW then reads x_buf/d_buf directly.
- dX output chain: conv11.dX unpack (1.93, 195 ch NCHW) -> cat-backward narrow -> LReLU/BN backward of upconv1 -> pack dY (1.34 incl dy16 copy)
  -> upconv1.dX unpack (2.31, 256 ch @36x36) -> upsample_nearest2d_backward (CPU 2x2 sum, reads 26.5 MB) -> cat backward -> BN/LReLU
  backward of upconv2 -> pack 0.26 -> upconv2.dX unpack 0.45 (256 ch @18x18) -> upsample backward -> cat backward -> ResNet (CPU).
  Redundant pairs: conv11.dX unpack + upconv1 dY pack = 1.93+1.34 = 3.3 ms; upconv1.dX unpack + upconv2 pack = 2.31+0.26 = 2.6 ms
  (the real waste there is the 36x36 full-res 256-ch intermediate, see 2b).
- The dX also computes the 3 image channels of conv11's input gradient (cout195 = 128+64+3), not needed, but it stays in block 2 either way (67 valid). ~0.

dW itself: 6 passes "dw run+sync" about 14 ms/step of NPU time (4.2/2.5/2.7/2.8 at 36x36, 0.87/0.87 at 18x18 in later steps), in a
separate context (D:2330-2345). It is fully hidden by the async helper (optimizer_ms 2.0 in tr_bfp; drv_async0 14.4 -> drv_async1 1.7,
total 125.7 -> 109.3).

## 2. Direct layout reuse
Geometry (P:870-924 vs P:828-865). For S==1 (all convs here): output pixel (b,y,x) sits at raw row (b*(H+2)+y)*wp + x, wp=W+2;
input pixel sits at (b*(H+2)+y+1)*wp + 1 + x. So out row + (wp+1) = in row: the same padded grid, a constant shift. In the
input, halo cells are zero (zero-filled once per shape, P:~800 contract); in the output the pad columns and between-image rows hold garbage
and would have to be masked or re-zeroed. Raw output is pixel-major with 128 bf16 contiguous per pixel (256 B); the bfp input is
[group][chunk(64ch)][8 blocks of 8 ch][2128 rows][9 B]. Re-encoding from the raw rows needs NO transpose (8-channel groups are already contiguous),
while the NCHW pack does a gather+8x8 transpose (gather_pixels, P:325) and the unpack the reverse (scatter_channels, P:334).

(a) Raw-to-raw re-encode (skip NCHW both ways) would need these ops applied in padded-NHWC raw layout:
BN (training stats: a pass over raw rows to get sum/sumsq per channel, then affine), LeakyReLU, nearest upsample (index shift
(y>>1,x>>1) while encoding), cat with the skip tensors (these are CPU NCHW tensors from conv0/1 and the input, so they would still
need a pack, but those could be packed once per step and cached), and the NCHW outputs autograd saves
(BN input and BN/LReLU output). All of these are CPU today. It needs a fused autograd Function; it saves roughly
the fwd unpacks 0.24+0.90+(0.53) and the transposes in the next pack. Big change, uncertain.

(b) Cheap partial version, local to pack/unpack, with no autograd change:
 - Fuse nearest-2x upsample into the pack: gather_pixels from the pre-upsample cat tensor with src index (y>>1)*(W/2)+(x>>1).
   Removes the upsample op's 26.5 MB write + read for upconv1 and 6.6 MB for upconv2 (upsample timing is not in these logs; unmeasured).
 - Fuse upsample-backward into the dX unpack: sum 2x2 pixels in fp32 while scattering, write the 18x18 tensor (6.6 MB for upconv1, vs 26.5 MB
   now) and delete the separate upsample_backward pass. It also cuts the 2.31 ms unpack.
 - Fuse BN statistics (per-channel sum, sumsq) into the fwd unpack: 128 accumulators, nearly free. BN then only does the apply pass.

## 3. Async / overlap
- run() pipeline (D:2113-2190): tasks ordered nb-major, pieces inner (4 pieces for g>=16, 2 for g>=6, else 1; piece_count D:1996). Loop:
  pack(j+1) [only for nb==0] -> wait(j) -> start(j+1) -> unpack(j). So the host does pack(j+1) while the NPU runs j, and unpack(j) while it
  runs j+1. But everything is on ONE host thread (with an OMP team inside), serial: exposed are piece-0 pack
  (~1/4 of 2.4-2.8 ms ≈ 0.6-0.7 ms) and the last-piece unpack (≈1/4 of unpack ≈ 0.2-0.6 ms) per call. In steady state the NPU is faster
  than the host (wait 0.12-0.4), so total ≈ pack + unpack + sync + ~1 overhead: host-bound. Overlap with the NPU is already
  as good as it gets; only reducing host work helps.
- For cout>128 (dX, nb=2) pack happens only in the nb=0 sweep, but nb=1's 4 piece-unpacks interleave with the NPU of nb=1 as well; fine.
- Async dW (DeferredJobLauncher D:345, AsyncDwPolicy D:414): the K-th NPU dX (last in backward) kicks a helper thread that runs the 6 dW
  dispatches (mostly blocking in shim_run_kernel) while the CPU does the remaining backward (ResNet, conv0-2 etc.). It works: optimizer
  14.4 -> 1.7 ms. It does not overlap anything with pack/unpack, and it forces a context switch afterwards (see wait 3 ms above).
- CPU dW overlap (D:4257-4285, XDNA_DISABLE_CPU_DW_OVERLAP): only taken when NPU dX exists but NO NPU dW plan. In NIS all 3 NPU dX convs
  have an NPU dW (w3_dw_plan set first, `else if`), so this path is effectively dead for the traced run (not verified at runtime).
  When it does run it uses a std::async thread calling sync_host_threads -> set_num_threads(10), i.e. 10 more OMP threads against the main thread's 10.
- Threads: g_pack_threads defaults 12 (XDNA_CONV_PACK_THREADS, D:2619-2625), but every run() resets it to g_host_threads
  (D:2080, 2392, 2419) = at::get_num_threads() of the owner thread (10 in the probe, D:129-155). nproc = 20. pack/unpack use
  num_threads(g_pack_threads) per loop (P:33,...,905), so they share GOMP's pool with PyTorch's intra-op ops when both are libgomp
  (-fopenmp in _build.py:54). The main thread is the only caller; the helper dW thread does no packing when buffers are reused, so
  contention is mainly the CPU-dW std::async case (dead) and the dW-job thread's pack_operands fallback. Thread scaling is poor:
  drv_t6/8/10/12/16 total = 139.4/132.0/130.3/138.4/143.3 ms, so 10 is the optimum, and more threads (shared with ATen ops and the NPU DMA) hurt. These loops are
  not memory-bandwidth-saturating (pack ≈ 25 GB/s, unpack 12-15 GB/s of write), so they are latency/instruction bound.

## 4. Why the dX unpack (2.1-2.3 ms) costs 2.5x the forward cout128 unpack (0.9 ms)
Mostly size, not an anomaly: dX of the cin128 conv has p.cout=256, so n_blocks = 2 (D:2064) and the unpack runs a full pass per 128-channel
block, writing a 256x36x36x40 bf16 = 26.5 MB NCHW tensor vs 13.3 MB for the forward (cout128: one block). Per block: 2.31/2 = 1.15 ms vs 0.90
(1.28x). The same applies to the other rows: fwd cout64 (cin195) writes 6.6 MB in 0.53 ms; dX cout195 writes 20 MB in 1.93 ms (128 + 67 channels). Overall ≈ 11-15 GB/s of NCHW write.
Why the rate itself is low (P:870-924, 334-346): scatter_channels does 8x8 transposes whose output is 16 B per destination channel plane.
Each 8-pixel x tile touches 8 different channel planes (stride H*W*2 = 2592 B) with 16 B stores, so each destination cache line is completed over
4-5 separate iterations and the write-allocate (RFO) traffic dominates; there is no non-temporal streaming. A row of W=36 pixels is
4 full 8-tiles + 4 tail pixels, and the tail goes through the scalar loop (1 pixel x all channels, 2-byte stores at stride H*W) = 11% of pixels,
probably 30-40% of the time (W=18: 2 tiles + 2 tail, same share). The 67-valid-channel block also has a 3-channel scalar remainder. In dX the second block's unpack
overlaps the NPU writing the next piece into the same LPDDR, and `sync` is 2x (two blocks, 0.30 vs 0.15). These are hypotheses
from reading the code; there is no microbenchmark.

## Ranked changes (ms/step estimates are from the tables above; confidence in parentheses)
1. Hide the dW->fwd hardware-context switch: after the last dW dispatch in the helper thread, immediately dispatch a tiny conv3x3w run
   (1 group, existing stream) so the xclbin switch happens during the optimizer/CPU forward instead of at the first fwd conv.
   Saves up to 2.5-3.0 ms of the 3.04 ms wait (medium; I did not confirm the cause, and the switch could land on other NPU users in between).
2. Fuse upsample-backward 2x2 sum into the dX unpack (upconv1: 26.5 -> 6.6 MB write, plus removes the CPU upsample_backward op).
   Saves ~1.5-2 ms on the unpack plus the removed op, estimated 2-3.5 ms total (medium; upsample_backward time not in logs).
3. Fuse nearest upsample into pack (read the cat'd 18x18/9x9 tensor with shifted index, skip the 26.5 MB + 6.6 MB materialization).
   Pack itself is unchanged, so the saving is the removed upsample op, estimate 1.5-3 ms (low-medium; op time unmeasured, doc claims
   "additional concat/upsample work").
4. Speed up unpack: handle the pixel tail (4 of 36, 2 of 18) with a masked/partial transpose instead of the scalar loop, and
   write full 64 B lines (transpose 16 pixels x 8 channels, or process a whole 36-pixel row per channel plane in registers, then store the row
   contiguously). Optionally non-temporal stores. Estimate 6.4 ms -> 4 ms: 1.5-2.5 ms (medium; hypothesis not benchmarked).
5. Skip x's unneeded 3 image channels and the padded garbage work in dX conv11: no real saving (same 2 blocks). ~0.
6. Fuse BN statistics into unpack and apply+LReLU+cat+bfp encode in the next pack (Function for conv-BN-LReLU): removes fwd BN stats
   and one NCHW pass, estimate 3-5 ms total of fwd/bwd (the doc puts BN+LReLU at ~11 ms/step) but a large autograd-correctness change
   (low confidence, high effort).
7. Look at the ~4.5 ms/step not covered by wait/pack/unpack/sync in the six traces (out alloc, ensure_host_current, weight slots, shim start):
   add trace buckets first. Potential 1-2 ms (low, unmeasured).
8. A second host thread for pack(j+1) concurrent with unpack(j-1): no gain expected (host-bound, thread scaling already negative past 10; bandwidth shared). <0.5 ms (low).
