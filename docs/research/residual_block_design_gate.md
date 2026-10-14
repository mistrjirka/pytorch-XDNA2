# Residual training block on the NPU: design gate (2026-10-13, the BFP-input experiment)

Status: analysis only, nothing built.  All "bounds" below are arithmetic on measured
numbers and are labelled as estimates; none is a promised saving.  Prewarm is parked
(opt-in, default off, no confirmed gain).

## The block (StyleNet `ResBlock`, 128 ch @ 9x9, B40, bf16)

```
c1 = conv1(x)          b1 = BN1(c1)         a1 = leaky(b1)
c2 = conv2(a1)         b2 = BN2(c2)         out = leaky(b2 + x)
```
Backward needs: `dout` -> leaky mask (out>0 <=> pre>0 for slope>0) -> `dpre` (feeds BN2 and the skip) ->
BN2 reductions (sum dy, sum dy*xhat over N,H,W) -> `dc2` -> conv2 dX (`da1`) and conv2 dW (needs `a1`) ->
leaky mask (a1>0 <=> b1>0) -> BN1 reductions -> `dc1` -> conv1 dX (+ skip `dpre`) and conv1 dW (needs `x`).
Training BN: biased batch variance for normalisation, unbiased for the running update, once per forward.

## Measured facts this design has to respect

| Fact | Value | Source |
|---|---|---|
| Residual stack alone, mode 2 | fwd 6.4 + bwd 13.7 = 20 ms (of ~93 ms step) | `docs/research/regions_vgg_pilot.md` |
| Of that: oneDNN conv | 5.0 fwd + 11.9 bwd; BN + leaky + add ~2.5 ms | same |
| One 9x9 conv on the NPU (forced), fwd or dX | total 0.34-0.37 ms: wait 0.21-0.23, pack 0.07, unpack 0.03 | `XDNA_CONV3X3W_TRACE`, this session |
| Same conv on CPU | ~0.5 ms fwd, ~1.2 ms fwd+bwd (dX and dW in one oneDNN call) | profile |
| Forced NPU for all 10 convs | fwd 5.8 (-0.5 ms), bwd 26.7 (+13 ms) | this session |
| NPU dW for 9x9 | not dispatched at the default floor; with the floor lowered the stack backward is 50 ms | this session |
| Program switch (fwd/dX <-> dW) | ~2.6-3.2 ms each way | trace, `style_step_breakdown.md` |
| Decoder dW on the NPU (async, helper thread) | 14.8 ms per step, hidden under a 24-26 ms CPU backward tail | trace |

The wait of 0.21-0.23 ms is almost all launch/DMA setup of a 3-group dispatch, not arithmetic (0.48 GMAC at
~10 GMAC/ms is 0.05 ms).  So the per-dispatch floor is about **0.25 ms**, and any design is counted in dispatches.

## Live-buffer budget (training needs more than "one activation fits")

One activation is 0.83 MB.  What backward needs per block (own private layout, bf16 unless noted):
`x` (= previous block's `out`), `c1`, `a1` (or its sign), `c2`, per-channel mean/invstd (fp32, 128 each).
Across five blocks: `out0..out5` (6) + `c1,a1,c2` per block (15) = **21 tensors = 17.4 MB**, against ~4 MiB of
memory-tile storage, before weights (0.3 MB bf16, ~0.17 MB bfp per conv), gradients, double buffers and halo padding.
So the stack is never on-chip: activations live in DRAM in a private layout and every stage streams through
the memory tiles.  "Resident" can only mean "no NCHW round trip and no CPU pass", not "never leaves the chip".
Recompute options (drop `a1`, rebuild it from `c1` + stats in the dW stage) trade DRAM for ALU and need the same
stats; they are a design choice to price, not a free win.

## Where the statistics could be computed (compare, none measured)

| Option | Idea | Cost / risk |
|---|---|---|
| A. Phased reuse of the conv cores | conv epilogue writes per-core fp32 partials (`[2][128]` = 1 KB/core, 32 KB per dispatch, masked for padded pixels); the same cores then run a reduce + broadcast phase, then the apply phase | extra phase/locks inside one program; no 33rd core needed; not prototyped |
| B. Host finishes the moments | partials go to a small host buffer; the CPU combines 32x256 floats (fp64) and passes mean/invstd as RTP/args to the apply step | one extra host round trip per BN (~0.1-0.2 ms floor) which is the same order as the dispatch floor |
| C. Reserved reducer / channel-sharded layout | different partition of channels vs pixels so one core owns a channel's whole reduction | changes the conv data layout; large redesign |

The bounded correctness milestone suggested by the review (stage 1: masked per-core partials in the conv epilogue,
host finish, checked against `batch_norm(training=True)`) validates masking and moments; it saves little on its own
and should be priced as a prerequisite, not a speed-up.

## Whole-step cost model (the isolated 20 ms is not 20 removable ms)

Forward: per block 2 dispatches minimum.  If BN/activation apply is folded into the next conv's input stage and the
stats come from the epilogue: 2 x 0.25 = 0.5 ms/block against ~1.3 ms on the CPU.  Upper bound **~0.8 ms x 5 = 4 ms**.
(Measured today with convs on the NPU and BN/activation on the CPU: only 0.5 ms.)

Backward: per block at least 2 dX dispatches + 2 dW dispatches + BN-backward reductions (need `x`/`c` on the dX core,
the harder part).  The dW program is a different xclbin: running it per conv costs a ~2.6 ms switch each way, so the
dWs must be batched at the end of backward (as the decoder dW already is).  Optimistic per block ~1.0-1.5 ms against
2.7 ms on the CPU.

Interaction with the existing async dW: today the CPU residual/encoder backward (~24-26 ms) hides 14.8 ms of NPU
decoder dW.  If residual backward moves to the NPU, the CPU tail shrinks and the residual dWs (10 more dispatches,
~3-5 ms) join the decoder dWs on the same NPU: the tail becomes bounded by the dW chain (~19 ms) rather than the
CPU (~24 ms).  So the realistic backward saving is about **24 - 19 = ~5 ms**, not 13.7 ms.

**Estimated whole-step ceiling for this project: ~8-10 ms (about 9% of the 93 ms step), if the stats, the backward
reductions and the scheduling all work.**  Multi-week kernel effort (new epilogue, reduce phase, apply stage, backward
BN reductions, private layout, ownership/lifetime), high risk.

## Candidates compared (bounds are estimates)

| Candidate | Remaining cost it attacks | Estimated ceiling | Effort / risk |
|---|---|---|---|
| Residual block on the NPU (this document) | residual stack 20 ms | ~8-10 ms | very high / high |
| Host-fused BN + LeakyReLU (+ skip add, resize/cat folding) as regions, forward and backward | BN 7.0 + activation 3.9 + cat/resize 5.5 = ~16 ms of CPU passes | ~6-8 ms | medium / low (same technique that gave -13 ms on the VGG loss; no hardware change) |
| NPU-resident VGG chain (taps and loss stay in NPU layout) | VGG convs ~13 ms CPU + pack/unpack | ~5 ms | high / medium |
| Pack/unpack host kernels (inline encode, unpack tails, streaming stores) | ~15 ms host conversion per step | ~3-5 ms | medium / low |
| Pool forward vectorisation | 2 x ~0.7 ms | ~1.4 ms | low / low |

## Recommendation

1. Do not start the NPU residual block as the next kernel project: its ceiling (~8-10 ms) is not larger than the
   host-fused BN/activation regions (~6-8 ms) which cost a fraction of the effort and risk, and which build exactly the
   pieces the NPU block needs anyway (training-BN semantics, masks, running statistics, saved-state contract, a
   numerical gate against `batch_norm`).
2. Order: host-fused BN + activation regions (forward and backward, module-pattern adapter first) ->
   statistics-probe prototype (stage 1 above, correctness only) -> then decide the one-block NPU prototype with real
   BN/activation costs in hand.
3. Gate for any NPU block prototype: forward + backward of ONE block matching input / weight / BN gradients and running
   state against an fp32 reference, then chained blocks, then a different legal shape, then full-step time with async
   dW running (not block time).
4. Also done in this pass: backward state of `frozen_feature_mse` now goes through `save_for_backward`; the placer keys
   include the input shape and are bounded (tests added).
