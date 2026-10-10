# XDNA2 PyTorch Training Backend — Agent Handoff
**Verified 2026-10-10, Europe/Prague.** This is the working-state handoff for the next ChatGPT/Pi/Codex agent. Reverify the repository status and measurements before treating them as current.

## 0. Mission and definition of success

Build a **general-purpose, correct, faster PyTorch training backend for AMD XDNA2** (Ryzen AI), with a real first-class torch.device("xdna"), rather than a special-case patch for the NIS coursework. Exploit XDNA2 when the *entire operation or graph region* benefits, keep cheap/fallback work efficient on CPU, and never declare a win based only on NPU kernel time. The practical reference is **native CPU BF16 training**; using CPU FP32 would set a misleadingly low bar.

The maintained backend is **still slower than CPU BF16** on the current style-transfer example. The accepted experimental NCHW-output optimization improves the XDNA backend by about 15–19% but does **not** close the roughly 2× gap. Do not misstate the project as finished.

## 1. Start here: which repo is authoritative?

| Role | Absolute path on mini | Notes |
| --- | --- | --- |
| **Maintained working source** | /home/jirka/programy/torch-xdna2 | Git origin git@github.com:mistrjirka/pytorch-XDNA2.git; package torch-xdna2. **Edit this repo for maintained changes.** |
| Core first-class device/C++ dispatch | /home/jirka/programy/torch-xdna2/xdna_train/_native/torch_xdna_device.cpp | PrivateUse1, XRT BO storage, CPU fallback, GEMM, Conv forward/dX/dW, native PyTorch registrations. |
| Runtime initialization and feature flags | /home/jirka/programy/torch-xdna2/xdna_train/device.py | Registers NPU artifact families, selects XCLBINs, configures allocator and backend. |
| Native compile/cache | /home/jirka/programy/torch-xdna2/xdna_train/_build.py | Compiles C++ against the **active exact PyTorch ABI**, caches under ~/.cache/torch-xdna2/native/. |
| Kernel packing/layout helpers | /home/jirka/programy/torch-xdna2/xdna_train/_native/conv_pack.cpp | CPU↔NPU memory layouts/packing; avoid repeated packing or excessive OpenMP threads. |
| Kernel programs / instruction binaries | /home/jirka/programy/torch-xdna2/xdna_train/_artifacts/ | 3x3 patch40, fullframe, GEMM, q4 resident artifacts. Keep program residency costs in mind. |
| Training motif coverage | /home/jirka/programy/torch-xdna2/xdna_train/training_coverage.py | MLP, residual CNN, U-Net, Transformer, Mobile/depthwise, embedding, nis-mini. |
| Tests | /home/jirka/programy/torch-xdna2/tests/ | tests/test_device_physical.py, tests/test_conv_output_layout.py, tests/run_conv_output_layout.py, tests/test_import.py. |
| **R&D/benchmark workspace** | /home/jirka/programy/torch-xdna-train | Older exploratory first-class/hybrid implementations, generators, benchmark scripts, experimental forks and saved results. **Not the maintained installable source.** |
| Experiment variants | /home/jirka/programy/torch-xdna-train/experiments/ | xdna2-layout-nchw/, xdna2-bn-consumer/, xdna2-native-fallback/, ABBA runner scripts. Do not mistake these for production. |
| NIS real workload | /home/jirka/programovani/magistr/nis/lab02_cnn | Original style_transfer_test.py, models.py, train.py, data.py and real image dataset. **Do not edit this coursework just to improve a benchmark.** |
| NIS Python and currently installed package | /home/jirka/programovani/magistr/nis/.venv/bin/python | Python 3.12.13, torch 2.14.1+rocm7.2. The installed xdna_train at .venv/lib/python3.12/site-packages/xdna_train is **not automatically the same** as the maintained checkout. |
| Lower-level research/runtime | /home/jirka/programy/xdna-engine | XRT/AIE program research and generators; other active, uncommitted work exists here. |

As verified on 2026-10-10:
- Maintained repository HEAD: 8b1fc6f (2026-10-04), with **uncommitted** work. Run git status before editing; avoid reset/clean/stash of other people's state.
- The maintained source has a **new opt-in** XDNA_CONV_OUTPUT_NCHW=1 implementation in _native/torch_xdna_device.cpp and an XDNA_ENABLE_PATCH_CONV=0 diagnostic registration switch in device.py, plus tests/README. They are **not** installed into site-packages and have **not** been committed.
- The R&D repository is a distinct, older checkout (HEAD 2e1b0d2, 2026-10-02) with extensive user work-in-progress. Do not overwrite its native files.
- NIS models.py has conv1 stride **2** restored; changing it to stride 1 causes a skip-connection/decoder shape mismatch even on CPU.
- NIS source hashes when benchmarks were captured:
    models.py: d4076596e4d34343ff0193baa8e79dee6adf4a0a1802a7eb9db5e79187d72f90
    style_transfer_test.py: 85345eeb69c8d0f1103cd24e67a1444187f3486890ca1f67f7ec6f0c61092bcc

## 2. Access, exact interpreter and import pitfall

ChatGPT can use the connected **Connect to mini** computer (online project "master"). Discover the current live session rather than relying on an old session ID. Host sandbox commands run on the real mini; keep writes inside explicit project paths. No need to hand off to another environment for basic measurements.

The host also has Pi at /home/jirka/.local/bin/pi, uv at /usr/bin/uv, git and rg. For long-running jobs use host job IDs and inspect their actual result/exit; do not infer a successful build from a cached log or a queued job.

**Critical import pitfall:** Python imports local xdna_train from the current working directory (or PYTHONPATH) before the installed distribution. To test maintained code unambiguously, launch from /tmp, or from the NIS working directory with PYTHONPATH selecting the maintained checkout. In diagnostic Python scripts, print xdna_train.__file__ and xdna_train.native_library_path().

Example:

    cd /tmp
    PYTHONPATH=/home/jirka/programy/torch-xdna2 \
    XDNA_CONV_PACK_THREADS=4 XDNA_BO_CACHE_MB=1024 \
    /home/jirka/programovani/magistr/nis/.venv/bin/python - <<'PY'
    import torch, xdna_train
    print(torch.__version__, xdna_train.__file__)
    print("xdna available:", xdna_train.is_available())
    print("native extension:", xdna_train.native_library_path())
    PY

The first import can compile C++ and take time; build fingerprints generate distinct .so cache directories. Do not edit installed site-packages to test candidate code. Do not launch overlapping hardware benchmarks because they compete for XDNA device context, shared fabric, memory and CPU.

## 3. User's relevant agent skills and instructions (verified)

The **Pi skills directory** is /home/jirka/.pi/agent/skills. Entries are symlinks into:
    /home/jirka/programovani/wiki/.wiki-pi/generated/async/skills/<skill-name>/SKILL.md

The following were actually present and their SKILL.md contents inspected:

| Skill | Use for this project |
| --- | --- |
| diagnosing-bug | First choice for mysterious slowdown, shape/correctness bug, fallback mismatch, flaky benchmark. Reproduce real behavior, test competing hypotheses, instrument narrowly, fix root cause. |
| running-real-data-smoke-tests | Validate the *actual* NIS loop, physical NPU, real input/targets and output, rather than synthetic/mocked operation tests only. Use isolated outputs. |
| review-final-diff | Focused check of changed C++/Python boundaries, memory ownership, saved-tensor lifetime, guards, tests and regression evidence before merging. |
| impl-check | **Deep**, high-risk cross-layer review when appropriate. Pi's skill explicitly requires independent behavior/contracts/design/runtime reviewer roles; do not claim impl-check passed without actually running all roles. |
| investigate-feasibility | Check version-sensitive AMD XRT / amdxdna / PyTorch 2.14 PrivateUse1 semantics against official docs/upstream source before relying on them. |
| pr-preparation | Git/worktree, packaging, new artifacts/tests and release-readiness audit if proposing to publish a wheel or commit. |

Other skills are available in the same folder (e.g. planning-qualitative-optimization and executing-qualitative-optimization), but **those address LLM-quality experiments, not physical NPU performance**. Avoid applying them indiscriminately.

Codex system skills are under /home/jirka/.codex/skills/.system, e.g. review-agent, skill-creator, openai-docs. They are not substitutes for direct physical NPU measurements. Pi's enableSkillCommands setting was observed as enabled, but read each skill's SKILL.md and use the active agent's supported invocation scheme.

This repository formerly had **no** AGENTS.md. The root AGENTS.md now points here to make the handoff discoverable. For architecture and prior experiments read:
- Maintained repo: README.md, docs/xdna-device-architecture.md, docs/fullframe-conv-backward-plan.md.
- R&D: docs/xdna-training-optimization-20261010.md and benchmark scripts under scripts/ and experiments/.
- Do not rely on historical ChatGPT cross-conversation verbatim retrieval as your only source of truth; repository artifacts/benchmark JSON are authoritative.

## 4. The reproducible, real-data training benchmark

Current target script:
    /home/jirka/programovani/magistr/nis/lab02_cnn/style_transfer_test.py

Network: MyNet(3,3,7), ResNet encoder/decoder with skip connections; B=40, RGB 36×36 patches, real aligned source/target frame, VGG19 perceptual+L1 loss, Adam, CPU BF16 or XDNA BF16. Real input directory:
    /home/jirka/programovani/magistr/nis/lab02_cnn/data/VID20261009180405/

**Original-script caveat:** its CPU mode uses **FP32**, not BF16. To measure the meaningful optimized CPU reference, explicitly select cpu-bf16 in the benchmark. It also loads **546 640×360 test frames twice** (input/target), needlessly pressuring 30-GiB RAM and 18-GiB swap during training. The benchmark avoids that by loading **one validation frame**, without altering real patch training, network, optimizer or loss. It intercepts the script in memory, uses a temporary output directory, omits final full-video render, and reports train-step median. It does **not** establish full-epoch video-export throughput.

Runner:
    /home/jirka/programy/torch-xdna-train/scripts/benchmark_original_style.py

**Validated one-round CPU-versus-NPU command:**

    cd /home/jirka/programy/torch-xdna-train
    PYTHONPATH=/home/jirka/programy/torch-xdna2 \
    XDNA_CONV_OUTPUT_NCHW=1 \
    XDNA_CONV_PACK_THREADS=4 \
    XDNA_BO_CACHE_MB=1024 \
    /home/jirka/programovani/magistr/nis/.venv/bin/python \
    scripts/benchmark_original_style.py \
      --modes cpu-bf16,xdna-bf16,cpu-fp32 \
      --slim-test --steps 12 --warmup 4 --threads 10 --rounds 2 \
      --out results/next-actual-script-abba

This harness runs backends **serially**, reversing order on round 2. Choose a fresh output folder to avoid overwriting earlier evidence. Use >=2 round orders before promoting changes. Check that it reports status "ok", finite/decreasing loss, acceptable gradients, and actual xdna execution rather than a misleading CPU-only fallback.

For the *unmodified* application end-to-end, run style_transfer_test.py only if you intend its full 200,000 patches, 546-frame video export and checkpoints; do not conflate that runtime with a 12-step benchmark. The source file and model may change independently; recheck SHA before A/B runs.

## 5. Latest measured checkpoints — physical mini, 2026-10-10

All timings below are **wall median training step** or mean of round medians as indicated, not NPU kernel-only throughput. Normal training benchmark instrumentation excludes expensive internal profiler tracing. Temperature, active desktop work, CPU threads and RAM contention affect repeatability. The first line of each result set contains actual run conditions.

**Most recent direct measurement** (4 warmups, 12 measured, 10 torch threads, real data and one test frame):

| Backend | Median ms/step | Throughput |
| --- | ---: | ---: |
| CPU BF16 | **233.0** | **4.29 it/s** |
| CPU FP32 | 514.2 | 1.94 it/s |
| Maintained XDNA BF16, producer NCHW | 519.1 | 1.93 it/s |

Saved evidence in:
- R&D/results/optimized-xdna-real-script-20261010/results.json
- R&D/results/optimized-xdna-matched-cpu-20261010/results.json

Here **R&D/results/** means /home/jirka/programy/torch-xdna-train/results/. The original script default CPU is FP32, so a claimed CPU 4 it/s is consistent with *CPU BF16*, not the script's default dtype.

**Reversed-order ABBA comparison with maintained fork** (average of 2 medians):

| Backend | ms/step |
| --- | ---: |
| CPU BF16 | 243.15 |
| XDNA native output strided [Y,X,B,C] | 598.32 |
| **XDNA producer-side contiguous NCHW** | **510.13** |

Source: R&D/results/maintained-fork-layout-ab-20261010/results.json.
The optimization gives ~14.7% faster XDNA training but is still ~2.1x slower than CPU BF16.

**Disable resident patch Conv? NO** (same script, two reversed-order rounds):

| Backend | ms/step |
| --- | ---: |
| XDNA + NPU resident patch Convs | **517.61** |
| XDNA with patch Convs forced to CPU-mapped fallback | 710.70 |
| CPU BF16 | 254.58 |

Source: R&D/results/maintained-fork-conv-gate-ab-20261010/results.json.
NPU patch kernels are **profitable**. Moving *all* work to mapped CPU on the XDNA device is worse than native CPU BF16.

**NCHW location A/B**:

| Layout path | ms/step |
| --- | ---: |
| Raw strided NPU output | 611.23 |
| Copy/contiguous at CPU BatchNorm consumer | 526.34 |
| **Copy/reorder once at NPU Conv producer** | **496.31** |
| Native CPU BF16 | 262.75 |

Source: R&D/results/bn-consumer-layout-ab-20261010/results.json.
Do not promote the consumer-side experiment: it was slower than producer-side in both rounds.

## 6. What code actually changed, and how to enable it

**Accepted experimental option, maintained source only**:

- native/xdna: /home/jirka/programy/torch-xdna2/xdna_train/_native/torch_xdna_device.cpp, inside XdnaConvRuntime::forward near the "GEMM rows are physically [Y,X,Batch,Cout]" comment.
- Set XDNA_CONV_OUTPUT_NCHW=1 to sync the raw NPU output to mapped host access, convert the logical [B,C,H,W] view to a contiguous NCHW XRT-backed tensor, mark it host-dirty and return it.
- Default **off**. A future XDNA-to-XDNA fused consumer might prefer the native layout, so blanket copies are not a general policy.
- device.py has XDNA_ENABLE_PATCH_CONV=0 **diagnostic only** (default enabled). The full-frame NPU programs are separate and were not the shape family under test.
- native Conv family is specifically B40 256→128, 3×3 stride1 pad1 at H=18 or H=36. Many other convolutions in MyNet still run through CPU fallback; do not call the backend universally accelerated yet.

**Physical correctness results**:
- tests/run_conv_output_layout.py passed with actual XDNA2 hardware at both H18 and H36: NPU Conv outputs are **bitwise equal** across layout variants; BN output and input/weight gradients satisfy CPU BF16 numerical tolerances.
- tests/test_conv_output_layout.py covers the same surface using pytest. In the NIS venv, the pytest package was absent when tested; use the standalone run_conv_output_layout.py instead or provision pytest explicitly without modifying other packages.
- Import-only and motif tests should be rerun after broader kernel changes.

The initial raw-strided Conv output made CPU BatchNorm on BF16 numerically **different** from CPU NCHW: on physical tests BN relative RMS error was ~0.89% (H18) and ~0.62% (H36); converted NCHW was ~0.0042% and ~0.0055%. That propagated to whole-network gradient summaries. Treat differences between raw and converted training gradients as a **numerics/layout effect**, not proof that conversion corrupts values. Test against CPU BF16.

**Validated physical smoke command:**

    cd /tmp
    PYTHONPATH=/home/jirka/programy/torch-xdna2 \
    XDNA_CONV_PACK_THREADS=4 XDNA_BO_CACHE_MB=1024 \
    /home/jirka/programovani/magistr/nis/.venv/bin/python \
    /home/jirka/programy/torch-xdna2/tests/run_conv_output_layout.py

## 7. Other concrete findings; do not repeat failed strategies

1. **XRT BO host mapping ≠ universally efficient CPU processing.** OneDNN BF16 convolutions using mapped XRT *input/output* memory were similar in isolated timings to normal CPU memory; no giant systematic XRT-memory bandwidth penalty was established. But noncontiguous NPU-style [Y,X,B,C] strides made CPU BatchNorm ~2x slower at 36×36 than NCHW, even though the view was zero-copy. One deliberate conversion can win end-to-end.
2. **Avoid unnecessary materialization.** Some native conv variants repack input/weights into staging BOs and sync them to NPU. Some backward fallback paths allocate CPU grad tensors then copy them into XDNA storage. Inspect actual XRT sync and coherency state before removing a sync; a host-only BO still has ordering requirements.
3. **Do not compute extra gradients.** The legacy CPU Conv backward "out" path requests dBias even when Conv has bias=False. An attempt to omit the third output caused PyTorch's out variant to throw "tensor does not have a device"; creating CPU outputs and copying back did not reliably improve end-to-end. This failed experiment was **reverted**. Any future fix must preserve native PyTorch API contracts and be tested end-to-end.
4. **Thread oversubscription.** Default Conv pack OpenMP count (12) conflicted with 10 Torch threads and desktop/game load. Experiments on the same machine favored XDNA_CONV_PACK_THREADS=4 (sometimes 8) under load. Do not hardcode 4 universally; consider runtime tuning and sensible caps. The CPU originally reached ~97–99°C during some runs. Paired ABBA and cooldown matter.
5. **Unsupported BO variants.** Normal/cacheable XRT user-pointer BOs failed on this driver, so do not naively replace host-only XRT storage with anonymous memory or assume share BOs work. Earlier 8-GiB monotonic XRT arena was **slower** than the reusable size-bin BO cache. Keep budgeted caching and look at graph-planned buffers only with evidence.
6. **Deferred CPU dW overlap** kept large saved tensors alive and regressed due to memory/lifetime pressure in previous full-frame experiments. Immediate NPU dX // CPU dW overlap can win for full-frame, but not automatically for small patches/loaded laptops.
7. **Original script's test preloading** produced a huge artificial slowdown on both CPU and XDNA: 546 full resolution frames loaded twice, ~3 GiB just for the tensors (plus workers, optimizer and BO cache). Benchmark --slim-test removes this outside the training computation, which is why older 1–2-second results should not be compared naively to the recent 0.2–0.5-second results.
8. **Fullframe vs patch geometry:** /home/jirka/programy/torch-xdna2/README.md documents q4 resident fullframe performance around 0.21 iterations/sec on a *different* large image workload (B4, not B40 style-transfer patches). Do not compare those iteration rates directly.

## 8. Next work with the highest expected value

The user explicitly wants a **universal NPU-accelerated training backend that doesn't slow down the CPU path**. Next agent should:

1. Reproduce the precise CPU BF16 vs XDNA baseline on a stable machine; report source hashes, device identity, package import, fallback counters, temperature and side workloads. Always measure full train-step including forward, backward, Adam and actual loss.
2. Trace the **backward hot path**: current real-script XDNA step ~519 ms, forward ~81.5 ms, VGG/L1 loss ~35.3 ms, backward **~396.6 ms**, Adam ~5 ms. Investigate CPU fallback conv dX+dW, BatchNorm backward, dW dispatch, repeated layout copies/contiguous buffers, extra dBias work and BO coherency. Avoid fixing one op based on theoretical FLOPs alone.
3. Expand **profitable NPU Conv support** beyond B40 256→128 H18/36 to shape families used by residual 128→128 H9 and encoder stride2/pointwise/decoder, while preserving PyTorch gradients and performance on unrelated models. Quantify pack + DMA + residency overhead before adopting each kernel.
4. Make a layout-cost decision by **consumer/graph region**, not merely a global environment flag. Consider fused/resident Conv→BN→activation or cost-aware CPU/NPU partitioning through torch.compile/AOTAutograd, while keeping cheap CPU regions in ordinary memory when feasible.
5. Add/maintain physical, numerical and full-network regression tests. Use training_coverage.py for CNN, U-Net, MLP, Transformer, Mobile/depthwise, embeddings, and nis-mini. Two 3-step motif timings do **not** justify a generic speed claim; use larger repeated tests where it matters.
6. Treat **CPU BF16 233 ms/step** as the initial pass/fail performance reference on this style-transfer case, not CPU FP32 ~514 ms. If a new XDNA path does not beat old XDNA without regressing correctness, reject it. The longer-term goal is faster than CPU BF16 at a useful range of models.
7. Review changes under the Pi skills above; preserve unrelated worktree modifications; only then discuss promotion/commit/package install with the user.

## 9. Files / exact run instructions for future agents

**Start by verifying paths rather than guessing:**

    git -C /home/jirka/programy/torch-xdna2 status --short
    git -C /home/jirka/programy/torch-xdna-train status --short
    rg -n 'XDNA_CONV_OUTPUT_NCHW|XDNA_ENABLE_PATCH_CONV' \
      /home/jirka/programy/torch-xdna2/xdna_train
    ls /home/jirka/.pi/agent/skills
    ls /home/jirka/programy/torch-xdna-train/results

**Reproduce full real-data A/B (serial in fresh processes):**

    cd /home/jirka/programy/torch-xdna-train
    /home/jirka/programovani/magistr/nis/.venv/bin/python -u \
      experiments/benchmark_fork_layout_ab.py

Results saved as results/maintained-fork-layout-ab-20261010/ by this historical script; consider cloning the runner/change output dir rather than overwriting existing evidence.

**Other useful actual benchmark runners** in R&D/experiments/:
- benchmark_patch_gate_ab.py — NPU resident patch Conv vs forced mapped CPU.
- benchmark_bn_consumer_ab.py — producer NCHW vs consumer-side BN repack.
- benchmark_real_nchw_ab.py — original package vs isolated NCHW patch variant.

**Other useful scripts** in R&D/scripts/:
- benchmark_original_style.py — real NIS script instrumented without edits.
- benchmark_training_matrix.py — general training motif matrix/CPU BF16/FP32/XDNA.
- profile_nis_phases.py — record high-level phase and fallback costs.
- benchmark_original_npu_tuning.py, benchmark_dx_dw_overlap_threads.py — pack threads and CPU/NPU overlap.
- validate_grad_mask.py — isolated hybrid-path backward mask validation; not proof of first-class device backward correctness.

**Native backend tests and results**:
    /home/jirka/programy/torch-xdna2/tests/
    /home/jirka/programy/torch-xdna-train/results/
    /home/jirka/programy/torch-xdna2/README.md

**Data provenance:** Bench runs generated temporary validation PNG/checkpoint and used the same NIS source/dataset. Native source and PyTorch site-packages were not replaced. There are no automatically scheduled future benchmark runs; do not claim a live watcher exists.

## 10. Do-not-destroy checklist

- DO NOT reset/clean, delete, reformat or silently commit the maintained, R&D, XRT engine or NIS worktrees. Several have **pre-existing uncommitted edits**.
- DO NOT modify NIS architecture/dataset/script just to get a better NPU benchmark. If changing application code is independently required, keep that as a separate explicitly reviewed change.
- DO NOT treat PYTHONPATH as proof that code loaded from the maintained checkout; print xdna_train.__file__ and native_library_path().
- DO NOT make unsynchronized host/device accesses; correctly retain mapped-storage source lifetimes across asynchronous CPU dW workers.
- DO NOT call a zero-copy graph automatically faster; consider downstream layouts, CPU path, and end-to-end scheduling.
- DO NOT declare CPU slowdown solved. As of this handoff, optimized XDNA BF16 ~519 ms/step and CPU BF16 ~233 ms/step on the targeted script.

**Suggested first action for the next agent:** connect to mini; read this file plus maintained README and both architecture docs; run git status and check the active imported package; reproduce a pair of real-data CPU BF16/XDNA BF16 steps; inspect mapped-CPU/backward hotspots; choose one falsifiable change with hardware numerical + ABBA evidence; leave unrelated work untouched.
