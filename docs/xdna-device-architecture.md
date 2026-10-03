# XDNA PyTorch Device Architecture

## Goal

The public interface should be ordinary PyTorch:

    import torch
    import xdna_train

    device = torch.device("xdna")
    model = model.to(device)
    x = x.to(device)
    target = target.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    y = model(x)
    loss = loss_fn(y, target)
    loss.backward()
    optimizer.step()

The XDNA backend must preserve normal PyTorch semantics while being at least as
fast as the existing direct-XRT experiments. The device abstraction is not a
requirement that every scalar operation execute on an AIE core. It defines one
shared XDNA memory/execution domain in which CPU and NPU are both eligible
engines.

## Non-negotiable performance lessons

The experiments in this repository established several constraints:

1. Reconfiguring/switching AIE programs can cost milliseconds and can dominate
   otherwise sub-millisecond kernels.
2. A mapped XRT BO exposed directly as PyTorch storage is measurably faster than
   copying NPU outputs into fresh CPU tensors.
3. Large rectangular GEMMs and selected NIS convolutions beat CPU when run as
   resident NPU programs.
4. Small/unfavourable operations should stay on CPU.
5. Packing and conservative host/device synchronization erase much of the raw
   NPU advantage.
6. Merely placing multiple kernels in one XRT hw_context does not guarantee that
   switching their AIE programs is cheap.
7. Graph-level scheduling is therefore required for peak performance.

Any first-class `xdna` device design that reintroduces per-op copies,
per-op BO allocation, or frequent AIE program changes is a regression even if
its API looks cleaner.

## Target architecture

    PyTorch
       |
       | torch.device("xdna")
       v
    PrivateUse1 device semantics
       |
       +-- XRT-backed StorageImpl / caching allocator
       |      |
       |      +-- mapped CPU pointer
       |      +-- xrt::bo handle
       |      +-- arena/sub-buffer metadata
       |      +-- coherency state
       |      +-- physical layout metadata
       |
       +-- eager compatibility
       |      |
       |      +-- resident NPU primitive when profitable
       |      +-- CPU implementation over the same mapped BO otherwise
       |
       +-- torch.compile / AOTAutograd performance path
              |
              +-- graph partitioning
              +-- layout assignment
              +-- liveness + arena allocation
              +-- cost model
              +-- fusion
              +-- AIE-program-aware scheduling
              +-- resident dispatch

CPU and NPU must operate on the same storage whenever possible.

## Storage model

The current PrivateUse1 prototype uses `posix_memalign`. That is only a
bring-up allocator. Replace it with storage that owns XRT resources.

Conceptually:

    struct XdnaAllocation {
        xrt::bo bo;
        void* mapped_ptr;
        size_t size;
        size_t arena_offset;
        CoherencyState coherency;
        LayoutId layout;
    };

The DataPtr context owns the allocation and releases it to a caching allocator
when the final PyTorch tensor/view disappears.

For steady-state training, prefer large persistent BO arenas and XRT native
sub-buffers over one BO allocation per logical tensor.

### Coherency

Track at minimum:

- CLEAN
- HOST_DIRTY
- DEVICE_DIRTY

Rules:

- NPU -> NPU: no host synchronization.
- CPU -> CPU: no device synchronization.
- CPU operation after NPU write: one sync-from-device.
- NPU operation after CPU write: one sync-to-device.
- explicit `.cpu()`: synchronize then copy only because the user requested a
  CPU storage object.

Do not unconditionally sync both directions around each operation.

## Universal buffer ABI

Current direct runtimes allocate BOs using `kernel.group_id(arg_index)`.
A general `torch.empty(device="xdna")` cannot know its future consumer.

General XDNA programs should converge on a universal tensor-memory ABI:
ordinary tensor buffers/arena slices live in a compatible memory group, while
control/instruction/scratch buffers are separate. Kernel arguments should carry
BOs plus offsets/layout descriptors rather than force each operator to allocate
new A/B/C buffers.

Acceptance criterion: an existing XDNA tensor can be passed to a supported NPU
kernel without copying or reallocating its payload.

## Layout model

Separate logical PyTorch shape/stride from physical XDNA layout.

Prefer ordinary PyTorch strides/storage offsets whenever a device-efficient
layout can be represented that way. Use backend-specific metadata only when
blocking or multiple cached representations cannot be represented by strides.

High-value cases:

- Conv weights should remain permanently in an NPU-friendly layout. Produce dW
  directly in that same layout so optimizer updates do not require repacking.
- Investigate HWBC-like activation storage that is both representable through
  PyTorch strides and efficient for Conv DMA.
- Halo padding can be backing-storage padding with the logical tensor viewing
  only the interior.
- nearest upsample followed by Conv should become source-addressing metadata,
  not a materialized upsample tensor.
- channel concatenation followed by Conv should become multiple source regions
  in the Conv schedule, not a copied concat tensor.

## CPU fallback policy

Generic PyTorch CPU fallback is useful for correctness but is not the desired
steady-state path. Since XDNA BOs are host-mapped, a CPU-selected operation
should execute on a CPU alias of the same storage, then update coherency
metadata.

The `xdna` device means XDNA memory/execution domain, not "AIE only".

This allows the cost model to choose, for example:

- large Conv/GEMM -> NPU
- small pointwise/reduction -> CPU
- Adam -> CPU initially
- fused graph region -> NPU

without moving tensors out of the XDNA device.

## Execution model

### Eager

Eager PrivateUse1 operators provide correctness and interactive usability.
Implement only operators needed for semantics or operations that are naturally
cheap/resident. Do not attempt to reproduce CUDA's huge eager kernel surface.

### Compiled

`torch.compile(..., backend="xdna")` is the performance path.

Compile forward and backward with AOTAutograd, then lower whole regions:

    graph
      -> partition
      -> layout assignment
      -> liveness
      -> arena assignment
      -> kernel/fusion selection
      -> AIE-program-aware schedule
      -> executable plan

The runtime should invoke an executable plan, not interpret FX nodes.

## Backward scheduling

AIE program switching is expensive. Use graph freedom to reduce it.

In particular, dX is on the critical dependency path while many dW calculations
are independent until the optimizer step. Where autograd dependencies permit,
propagate dX through a region first, then batch compatible dW work. This can
convert many fwd/dX <-> dW program switches into one phase transition per
training step.

The scheduler cost model must include resident-program state, not only
hw_context identity.

## NIS-specific fusion opportunities that should generalize

The NIS model repeatedly uses:

    Conv -> BatchNorm -> LeakyReLU

Training BN still needs reductions, but normalization/affine/activation can be
fused with surrounding device regions.

Also eliminate materialized:

- nearest-neighbor upsample before Conv;
- concat immediately consumed by Conv.

These are graph/layout optimizations, not NIS hard-coding.

## Optimizer

Keep Adam on CPU initially unless measurement proves otherwise. It is
FP32/pointwise/memory-heavy and the CPU can operate directly on host-mapped XRT
storage. This gives normal `optimizer.step()` semantics with no tensor
migration.

Later measure AIE implementations independently.

## Device implementation phases

### Phase 0: ABI-stable development device

- PrivateUse1 renamed to `xdna`.
- device module registered.
- allocation/copy/view/module.to semantics tested.
- native extension cached per Python/Torch ABI so multiple environments do not
  overwrite each other.

### Phase 1: real XRT storage

- replace host malloc with mapped XRT BO allocation;
- expose BO identity to the NPU runtime;
- caching allocator / arenas;
- views keep allocation lifetime;
- CPU aliases share the same bytes.

Gate: `torch.empty(device="xdna")` owns an XRT BO.

### Phase 2: zero-copy existing kernels

- make GEMM and Conv runtimes consume XDNA storage directly;
- eliminate A/B/C staging BOs when inputs/outputs are already XDNA tensors;
- add coherency tracking.

Gate: PrivateUse1 Conv/GEMM latency <= 1.05x direct-XRT latency.

### Phase 3: compiled regions

- compile XDNA graphs from AOTAutograd;
- persistent arena assignment;
- layout propagation;
- fusion;
- AIE-program-aware scheduling.

### Phase 4: NIS graph coverage

- Conv/BN/LeakyReLU regions;
- concat and nearest upsample elimination;
- BCE/reductions as appropriate;
- CPU Adam over shared XDNA storage initially.

## Performance gates

1. XDNA device Conv <= 1.05x the corresponding direct XRT Conv.
2. Rectangular Linear fwd+dX+dW <= about 2.3 ms, targeting the direct
   1.94-2.12 ms result.
3. Preserve >=1.22-1.27x speedup on the profitable NIS decoder Conv shapes.
4. Whole NIS patch-training step >1.15x CPU.
5. After residency/fusion, target >1.3x CPU, with 1.5x+ as a stretch goal.

Every optimization should be evaluated against end-to-end training as well as
microkernels.

## Packaging

The eventual package should build its native PrivateUse1 extension against the
exact PyTorch ABI in the target environment. Never ship or copy a development
extension built against a different Torch installation.

The notebook-facing API should require only normal device selection; explicit
`xdna_train.accelerate(model)` remains a compatibility path while the
first-class device backend matures.


### Rejected experiment: standard XRT user-pointer BOs

On the current amdxdna/XRT stack, wrapping ordinary 4 KiB-aligned host
allocations with the standard XRT user-pointer constructors is not usable.
Hardware probes returned EOPNOTSUPP for normal BOs, required the driver-
specific AMDXDNA_BO_SHARE type for cacheable BOs, and failed mmap for
host-only and xrt::ext::bo user-pointer variants. The public XRT constructors
tested do not expose a working AMDXDNA_BO_SHARE path here. Keep canonical
device storage on XRT-allocated BOs unless a supported shared-user-memory API
is found and independently benchmarked.


### Rejected experiment: one large XRT sub-buffer arena

An 8 GiB host-only XRT arena with page-aligned sub-buffer allocations was
implemented and validated for CPU round-trip and direct NPU GEMM. It removed
fresh BO allocations, but the exact full-frame epoch regressed to 0.09805 it/s
versus 0.10194 it/s for the 8 GiB size-bin BO cache. The 2,208 sub-buffer
create/free operations per epoch were not free enough to justify the arena.
The observed peak live device storage was about 4.81 GiB. Keep the reusable BO
cache for now; revisit a graph-planned arena only when lifetimes can be assigned
statically rather than creating XRT sub-buffer objects for every eager tensor.
