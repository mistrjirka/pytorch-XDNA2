# Full-frame XDNA Conv backward plan

## Goal

Accelerate the actual NIS full-frame training workload, not proxy patch geometry.
The target call is ordinary PyTorch:

    model.to("xdna")
    loss.backward()

PyTorch autograd remains the graph owner. The XDNA backend implements explicit
Conv backward primitives behind `aten::convolution_backward_overrideable`.

For a 3x3 stride-1 Conv:

- dX is another virtual-im2col Conv using dY and rotated/transposed weights.
- dW is a reduction GEMM: `patches(X).T @ dY`.
- dbias is a small reduction and can stay on CPU initially.

## Measured real-workload baselines

BF16, CPU, 16 threads:

| shape | dX | dW | combined |
| --- | ---: | ---: | ---: |
| B4 128->128 146x141 | ~41 ms | ~54 ms | ~95 ms |
| B4 256->128 292x282 | ~281 ms | ~272 ms | ~553 ms |
| B4 256->128 584x564 | ~1050 ms | ~1047 ms | ~2097 ms |
| B4 195->64 584x564 | ~550 ms | ~544 ms | ~1094 ms |

The 256->128 @ 584x564 decoder layer is the first target.

## Phase 1: dX584

Use the existing full-frame resident XCLBIN and a dedicated `dx584`
instruction stream.

Requirements:

- BF16 dY and weights.
- Accept channels-last XDNA dY directly.
- Pack dY halo once.
- Transform/rotate weights without materializing a transposed tensor.
- Produce 256 input-gradient channels as two 128-channel output slices.
- Reuse the packed dY across both slices.
- Return compact channels-last XDNA storage.
- No NCHW round-trip between autograd nodes.

Promotion gate:

- numerical agreement with CPU within the existing BF16 tolerance;
- complete dX path, including pack/sync/output conversion, < 700 ms;
- no regression in the real full-frame training step.

## Phase 2: dW584 direct-DMA

Logical GEMM:

    [9*Cin, B*H*W] @ [B*H*W, Cout] -> [9*Cin, Cout]

For 256->128 @ 584x564:

    M = 2304
    K = 1,317,504
    N = 128

Never materialize `patches(X).T` (roughly 6 GiB in BF16).

Extend the existing virtual A-transpose/direct-DMA dW generator:

- stream X patch taps directly from activation storage;
- stream dY in the same reduction order;
- BF16 multiply with FP32 accumulation;
- runtime K loop / bounded descriptors;
- retain an output tile in FP32 accumulators across K chunks where possible;
- write dW once, then expose FP32 or narrow to BF16 according to training mode.

Promotion gate:

- correctness against CPU;
- complete dW path < 700 ms;
- dX+dW together materially beat the ~2.10 s CPU backward for this layer.

## Phase 3: scheduling

Avoid alternating dX/dW program images per layer.

Preferred eager path:
- keep dX and dW in one compatible resident training image if resource limits allow.

Preferred compiled path:
- AOTAutograd backward graph;
- execute the dX dependency chain first;
- defer independent dW work until before optimizer.step();
- pay at most a coarse program transition instead of one transition per layer.

## Phase 4: extend by measured impact

After 584x564:

1. 256->128 @ 292x282
2. 195->64 @ 584x564 (pad channels internally if profitable)
3. repeated 128->128 @ 146x141 residual backward
4. stride-2 encoder backward
5. batch-2 forward family for validation

## XDNA2 anti-pattern constraints

- Do not materialize im2col matrices.
- Do not promote a primitive based only on kernel time; include pack/sync/layout.
- Do not force NCHW between producer and consumer.
- Do not create a different XRT/PDI transition for every backward sub-op.
- Do not keep enormous monotonic arenas alive merely to reduce BO allocation calls.
- Measure the exact DRIVE train/validation loop after every promoted family.

## Measured heterogeneous overlap result

The first whole-training scheduling optimization is now validated: when a Conv
backward has a profitable NPU dX but dW remains on CPU, run both concurrently
over the same host-visible XRT storage and join before returning from eager
autograd. This is enabled by default; `XDNA_DISABLE_CPU_DW_OVERLAP=1` is the
debug/AB control.

For B4 256->128 at 584x564, complete dX+dW timing (including XDNA layout and
synchronization) improved from ~1.55 s serial to ~1.04 s overlapped, with the
same BF16 numerical error as the serial path. This is close to the expected
`max(NPU dX, CPU dW)` bound.

Unprofiled exact NIS epoch ABBA-style results on the performance profile:

- serial XDNA: 0.121792 and 0.123024 it/s;
- overlapped XDNA: 0.131510 and 0.131079 it/s;
- recent CPU BF16 baseline: ~0.11297 it/s.

This changes the dW decision: an NPU dW kernel is not automatically useful if it
serializes behind dX on the same array. It must beat the heterogeneous overlap
or participate in a graph schedule that defers/batches dW without lengthening
the dX critical path.

## Rejected experiment: cross-layer deferred CPU dW

A second scheduling experiment returned an XDNA gradient immediately, queued CPU
dW on a bounded worker, and allowed autograd to advance the NPU dX chain before
joining the weight gradients. This was intentionally gated by
`XDNA_DEFER_CPU_DW=1`; the normal immediate `NPU dX || CPU dW` scheduler
remains the default.

Two correctness/lifetime issues had to be exposed before the experiment was
valid:

- a pending host write must be a real storage dependency, so CPU/device reads,
  device synchronization and BO reuse have to fence it;
- CPU aliases of mapped XDNA storage are non-owning, so queued work must retain
  the source/destination XDNA tensors until completion.

After those fixes the result was numerically correct, but performance regressed:

- immediate scheduler: 0.137675 it/s in the matched control;
- fully deferred/correct scheduler: 0.111656 it/s, with the fourth train batch
  reaching 10.96 s under memory pressure;
- bounded window=1: 0.120286 it/s, still slower than immediate.

The full-defer variant retained large saved activations/dY tensors long enough
to create severe memory pressure, an O6/resource-lifetime failure. Even a
one-job window lost because CPU dW then contended with the intervening
CPU-heavy backward graph for threads and memory bandwidth. This rejects eager
cross-layer dW deferral on the current shared-memory machine. It does not reject
a future compiler schedule that can reason about lifetimes, recomputation and
CPU/NPU bandwidth globally.

The important scheduling rule is therefore: overlap CPU dW with NPU dX inside
the same Conv where the CPU would otherwise be idle, but do not extend that
overlap across arbitrary CPU-heavy graph regions without a graph-level resource
model.
