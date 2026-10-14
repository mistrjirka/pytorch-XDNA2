# Modes audit: environment variables, modes and NPU families

## Automatic compiler setup

`XDNA_AUTO_SETUP=0` disables managed compiler installation; `UV_OFFLINE=1` prevents
first-time downloads. Neither disables an already completed managed compiler.
An explicit `XDNA_IRON_PYTHON` (including an empty value) overrides automatic
selection. `TORCH_XDNA2_CACHE` covers native artifacts and the managed compiler;
otherwise `XDG_CACHE_HOME` or `~/.cache` is used. These are setup controls, not
convolution performance modes. See [usage](usage.md#convolution-compiler).


Inventory of everything that selects a code path in `xdna_train`, taken at commit
`9e1a5de` (before pruning). `file:line` references are to that commit; the
**Status** column says what the pruning branch did with each row. The last section
lists the evidence and how it was gathered.

Verdicts: **KEEP** (core or well-placed knob), **KEEP-BUT-DOCUMENT** (useful, but
undocumented or only partly documented), **MERGE/SIMPLIFY** (keep the behaviour,
collapse the mechanism), **REMOVE** (dead, shadowed, superseded, or measured slower with
no recorded benefit).

Categories: core = default-on feature; tuning = numeric/behavioural knob; debug =
diagnostic switch or kill switch; experiment = opt-in, off by default; dead = cannot be
reached with the shipped artifacts or at default settings.

## 1. NPU families (what actually runs on the NPU)

Reachability was measured with temporary per-path counters (not committed) on the
workloads listed in section 4; "default" means default env, warm conv3x3w stream cache.

| Family | Shapes it handles | Reached at default? | Artifacts | Category | Evidence | Verdict |
|---|---|---|---|---|---|---|
| conv3x3w fwd + dX (`_designs/conv3x3w.py`) | 3x3/s1/p1, any batch/H/C, W tiled to <= 34 columns; cost model | **Yes**, all of style-transfer (upconv2/upconv1/conv11 fwd+dX), segmentation and style patch, all inference | built on demand into `~/.cache/torch-xdna2/conv3x3w/<hash>` (needs IRON python or a warm cache) | core | `OPTIMIZATION_LOG.md` (166 -> ~129 ms/step), `style_round4_ac.json`, tests | KEEP |
| conv3x3w_dw (NPU dW, deferred) | dW >= 1 GMAC | **Yes** (style-transfer: 256->128 @18/36, 195->64) | same cache | core | log: 171 -> 148 ms, bfp16 gate | KEEP |
| Patch conv (`conv_runtime_slot`, `XdnaConvRuntime`, 9e1a5de:1501-1950, 4884-4898, 5155-5165) | B40, 256->128, 18x18 and 36x36 only | **No**: shadowed by conv3x3w (`try_forward`/`plan` are tried first). 0 hits in style-transfer, segmentation and style patch, style7 | `conv3x3_patch40_128` (1.1 MB) | dead at default | Warm cache: style-transfer 131.5/138.2 ms with it vs 139/137 ms with `XDNA_ENABLE_PATCH_CONV=0`. Without conv3x3w (no IRON, the only case where it runs): 198.7/200.0 ms with it vs 190.7/190.9 ms with no NPU conv at all, i.e. it is now slower than the CPU conv (the 517 vs 710 ms README number predates the CPU-side fixes) | REMOVE |
| Full-frame legacy fwd (`fullframe_conv_runtime_slot`, fwd292 / fwd584) | B4, 256->128 @ 292x282 and 584x564 | **Yes** (segmentation full-frame) | `fullframe_strix_b4_m96` | core | README: 0.200 -> 0.2038 it/s with the q4 family; `XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_CONV=0`: +7 % step time here | KEEP |
| Full-frame legacy res146 / dX292 / dX584 streams (same runtime) | B4, 128->128 @146x141 fwd, dX @292/584 | **No**: the q4-resident family below is tried before them for exactly the same shapes | same dir (res146.bin, dx292.bin, dx584s73.bin, ~5.7 MB; allocates ~150 MB of staging BOs) | superseded | README: legacy 0.2008 vs q4 resident 0.2038 it/s; counters: all dX / res146 calls go to `fast_dx584` | REMOVE |
| Fast full-frame (q4 resident; `fast_dx584_runtime_slot`) | B4: res146 fwd, dX 292 and 584 stripes | **Yes** | `fullframe_q4_resident_k1152_m96` | core | README ABBA +2.8 %; `XDNA_ENABLE_FAST_FULLFRAME=0`: +7 % step time | KEEP-BUT-DOCUMENT (kill switch) |
| `configure_fast_dx584` (single-stream variant of the above) | dX584 only | No: only chosen when the fast artifact dir has no `res146.bin`; the bundled one has | none | dead | -- | REMOVE |
| Fast conv11 dX (`fast_conv11_runtime_slot`) | B4, 195->64 @ 584x564 dX | **Yes** | `conv11_dx_q4_resident_k576_m96` | core | README +3.6 %; `XDNA_ENABLE_FAST_CONV11=0`: +7 % step time | KEEP-BUT-DOCUMENT (kill switch) |
| Full-frame dW (`XdnaFullFrameDwRuntime`, 9e1a5de:3611-3809, 5187-5199, 5466-5469) | B4, 256->128 @ 584x564 | No: needs `XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_DW=1` **and** `_artifacts/fullframe_dw_unbundled`, which does not exist in the repo | missing | dead | initial commit only, no measurement anywhere | REMOVE |
| GEMM family (`gemm_runtime_slot`, `mm_xdna`) | `mm` of exactly 256x768x3072, 256x3072x768, 3072x256x768, bf16 contiguous | Only for those shapes (none of the style-transfer/lab/zoo workloads) | `train_gemm_64x32x96` | core (narrow) | micro-benchmark here: 0.64 vs 0.87 ms (fwd), 0.61 vs 0.70 (dX), 0.95 vs 0.85 (dW, with readback); `tests/run_gemm_safety.py`, `test_device_physical.py` | KEEP-BUT-DOCUMENT |
| Fused cat2 + upsample2 fwd (`forward_cat2_upsample2`) | B4 full-frame decoder, only when `XDNA_LAZY_DECODER_VALUES=1` produced a virtual value | No (opt-in) | uses the legacy full-frame stream | experiment | see lazy decoder values | REMOVE |
| CPU islands (`cpu_islands.py`) | `lstm.input`, `gru.input` | Yes | -- | core | log: LSTM 48 -> 18.5-20.8 ms | KEEP |
| Step autotuner / `npu_mode` 2-1-0 | whole model | Yes | -- | core | log: UNet 67 -> 54 ms | KEEP |

## 2. Environment variables

`device.py` lines are in `xdna_train/device.py`, `cpp` lines in
`xdna_train/_native/torch_xdna_device.cpp` (both at 9e1a5de).

| Variable | Default | What it does | Read at | Category | Evidence of value | Verdict | Status |
|---|---|---|---|---|---|---|---|
| `XDNA_AUTOGRAD_THREAD` | 0 | `1` keeps PyTorch's separate autograd device thread | device.py:188 | debug | log: MLP step 8.0 -> 3.8 ms with it off; README section | KEEP-BUT-DOCUMENT (already in README) | kept |
| `XDNA_AUTOTUNE` | 1 | `0` skips the step autotuner (mode 2) | device.py:585,592 | debug | log (UNet, ABBA) | KEEP | kept |
| `XDNA_AUTOTUNE_WARMUP` | 3 | steps before timing | device.py:595 | tuning | -- | KEEP | kept |
| `XDNA_AUTOTUNE_STEPS` | 2 | timed steps per block | device.py:596 | tuning | -- | KEEP | kept |
| `XDNA_AUTOTUNE_ROUNDS` | 1 | ABBA rounds | device.py:597 | tuning | -- | KEEP | kept |
| `XDNA_AUTOTUNE_TOLERANCE` | 0.03 | NPU must beat CPU-only by this | device.py:598 | tuning | -- | KEEP | kept |
| `XDNA_AUTOTUNE_VERBOSE` | 0 | print the choice | device.py:575 | debug | README | KEEP | kept |
| `XDNA_NPU_MODE` | unset | fix mode 0/1/2, disables tuner | device.py:588 | debug | used by `tests/quick_check.py`, benchmarks | KEEP | kept |
| `XDNA_BO_CACHE_MB` | 25 % RAM, 512..8192 | size of the freed-BO cache | device.py:82,99; cpp:224 | tuning | README (full-frame needs GiB-sized cache); `tests/test_device_physical.py` | KEEP-BUT-DOCUMENT | kept |
| `XDNA_BO_ARENA_MB` | 0 (off) | carve tensors from large sub-buffer arenas | cpp:231, 253-307 | experiment | `docs/xdna-device-architecture.md` "Rejected experiment: one large XRT sub-buffer arena": 0.09805 vs 0.10194 it/s; README design rule says not to do it | REMOVE | removed |
| `XDNA_TENSOR_BO_FLAG` | host_only | BO flag for tensor storage: host_only / cacheable / normal | cpp:194 | experiment | nothing records a benefit; docs say the other flags fail mmap or break coherency assumptions (flush logic assumes host-only BOs) | REMOVE | removed |
| `XDNA_HOST_THREADS` | follow caller | pin the intra-op width of host compute | cpp:144 | tuning | log: 6 ~ 10, 20 is 2x worse | KEEP-BUT-DOCUMENT | kept |
| `XDNA_CONV_PACK_THREADS` | 12 | OpenMP width of full-frame pack/unpack (read 4x) | cpp:1669,3093,3151,3652 | tuning | project notes (4 under load) | MERGE/SIMPLIFY (one helper) | merged |
| `XDNA_IRON_PYTHON` | auto-detect | python of the mlir-aie env; empty disables on-demand builds | device.py:113 | core config | README | KEEP | kept |
| `XDNA_CONV3X3W_DIR`, `XDNA_CONV3X3W_DW_DIR` (name built as `XDNA_{NAME}_DIR`) | cache dir | directory with xclbin + streams | device.py:152 | debug | used when testing other builds | KEEP-BUT-DOCUMENT | kept |
| `XDNA_CONV3X3W_MIN_MACS`, `XDNA_CONV3X3W_DW_MIN_MACS` (built as `XDNA_{NAME}_MIN_MACS`) | 20e6 / 3e9 | size floor | device.py:153 | tuning | README; tests set 0 | KEEP | kept |
| `XDNA_CONV3X3W_SYNC_BUILD` | 0 | build streams before the first call | device.py:153 | debug | README, tests | KEEP | kept |
| `XDNA_CONV3X3W` | 1 | `0` disables the family | device.py:219 | debug | README, `quick_check`, sweep | KEEP | kept |
| `XDNA_CONV3X3W_DW` | 1 | `0` keeps every dW on the CPU | device.py:228 | debug | README | KEEP | kept |
| `XDNA_ASYNC_DW` | 1 | `0` starts deferred NPU dW only at the first reader; `1` starts them on a helper thread after the K-th dX call once two steps agree on K (`xdna_train.async_dw_stats()`) | -- | core | style-transfer step 122.6 -> 110.5 ms (ABBA, AC, 2026-10-13); tests/run_async_dw.py | KEEP | new |
| `XDNA_CONV3X3W_FORCE` | 0 | bypass the cost model | device.py:222; cpp:2309 | debug | tests | KEEP | kept |
| `XDNA_CONV3X3W_NPU_OVERHEAD_MS`, `_NPU_GMACS`, `_HOST_GBS`, `_CPU_GMACS`, `_MARGIN` | 0.5, 5.5, 20, 0.9, 0.8 | cost-model constants | cpp:2313-2326 | tuning | log (cost model rows), `run_conv3x3w_costmodel.py` | KEEP | kept |
| `XDNA_CONV3X3W_PIECES` | auto (1/2/4) | pipelined piece count | cpp:2484 | tuning | log: pipelined dispatch 147 -> 129 ms | KEEP-BUT-DOCUMENT | kept |
| `XDNA_CONV3X3W_TRACE` | 0 | per-conv trace | cpp:2527,2913 | debug | log | KEEP | kept |
| `XDNA_CONV3X3W_WEIGHT_CACHE` | 1 | `0` disables the weight cache | cpp:2413 | debug | `tests/run_conv3x3w_torch.py` | KEEP | kept |
| `XDNA_ENABLE_PATCH_CONV` | 1 | register the patch conv family | device.py:207 | dead at default | see family table | REMOVE | removed |
| `XDNA_PATCH_CONV_ARTIFACT_DIR` | bundled | alternative artifact dir | device.py:202 | dead | -- | REMOVE | removed |
| `XDNA_CONV_OUTPUT_NCHW` | 1 | patch conv only: `0` returns the raw [Y,X,B,C] layout | cpp:1908 | dead (patch conv only) | project notes: NCHW is +15 % (598 -> 510 ms) | REMOVE | removed |
| `XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_CONV` | 1 | `0` disables the legacy full-frame fwd streams | device.py:239 | debug (misnamed "experimental") | README; `=0`: +7 % here | KEEP-BUT-DOCUMENT | kept |
| `XDNA_FULLFRAME_ARTIFACT_DIR` | bundled | alternative artifact dir | device.py:234 | debug | used to try new kernels | KEEP-BUT-DOCUMENT | kept |
| `XDNA_ENABLE_FAST_FULLFRAME` | 1 | `0` disables the q4 resident family | device.py:280 | debug | README, `=0`: +7 % | KEEP-BUT-DOCUMENT | kept |
| `XDNA_ENABLE_FAST_DX584` | 1 | old name of the previous | device.py:281 | dead alias | -- | REMOVE | removed |
| `XDNA_FAST_FULLFRAME_ARTIFACT_DIR` | bundled | alternative artifact dir | device.py:271 | debug | -- | KEEP-BUT-DOCUMENT | kept |
| `XDNA_FAST_DX584_ARTIFACT_DIR` | -- | old name of the previous | device.py:272 | dead alias | -- | REMOVE | removed |
| `XDNA_ENABLE_FAST_CONV11` | 1 | `0` disables the conv11 dX program | device.py:323 | debug | README, `=0`: +7 % | KEEP-BUT-DOCUMENT | kept |
| `XDNA_FAST_CONV11_ARTIFACT_DIR` | bundled | alternative artifact dir | device.py:318 | debug | -- | KEEP-BUT-DOCUMENT | kept |
| `XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_DW` | unset | enable full-frame NPU dW | device.py:353 | dead (no artifact) | none | REMOVE | removed |
| `XDNA_FULLFRAME_DW_ARTIFACT_DIR` | `fullframe_dw_unbundled` (absent) | artifact dir for it | device.py:348 | dead | -- | REMOVE | removed |
| `XDNA_FULLFRAME_TM` | from manifest | tile M, set by Python from `manifest.txt` into `os.environ` three times and read back by C++ | device.py:248-338; cpp:3100,3160 | tuning (internal channel) | all bundled artifacts say 96 | MERGE/SIMPLIFY (pass the manifest tile as an argument) | merged |
| `XDNA_FUSED_OPTIMIZER` | 1 | fused optimizer default | device.py:382,453 | core/debug | log: MLP 3.7-4.2 -> 2.9-3.5 ms | KEEP | kept |
| `XDNA_CPU_ISLANDS` | 1 | LSTM/GRU as oneDNN CPU kernels | device.py:371; cpu_islands.py:113 | core/debug | log: 48 -> 19 ms | KEEP | kept |
| `XDNA_CPU_DW_BATCH_CHUNK` | 1 for B4, else 0 | split CPU dW by batch chunk | cpp:5382 | tuning | README: +2.4 % ABBA; here `=0`: +4.5 % step time; `test_device_physical.py` | KEEP-BUT-DOCUMENT | kept |
| `XDNA_DISABLE_CPU_DW_OVERLAP` | 0 | `1` runs CPU dW after NPU dX | cpp:5218 | debug (A/B control of a core feature) | plan doc: 0.1218 -> 0.1315 it/s; here `=1`: +30 % | KEEP-BUT-DOCUMENT | kept |
| `XDNA_DEFER_CPU_DW` | 0 | queue CPU dW on a worker, return gradient storage immediately | cpp:5222 | experiment | `fullframe-conv-backward-plan.md` "Rejected experiment": 0.1117 vs 0.1377 it/s; log: "unproven", net 0-3 ms on style-transfer; here +50 % step time | REMOVE | removed |
| `XDNA_DEFER_CPU_DW_WINDOW` | 2 | queue depth of the worker | cpp:568 | experiment | same (window=1 also lost) | REMOVE | removed |
| `XDNA_HOLD_CPU_DW` | 1 (only matters with DEFER) | hold queued CPU dW until the NPU dW starts | cpp:582 | experiment | log only | REMOVE | removed |
| `XDNA_LAZY_DECODER_VALUES` | 0 | keep cat/upsample results virtual and fuse them into the next conv | cpp:1013,4176 | experiment | README: "gain smaller and more variable"; here `=1`: 3762/4245 vs 3660/4159 ms (+2 %, i.e. no gain); source of the recent set_(Storage)/view audit bugs | REMOVE | removed |
| `XDNA_LAZY_DECODER_DW` | 0 | CPU dW directly from the virtual decoder inputs | cpp:4997 | experiment | none; here (with lazy values) +8 % | REMOVE | removed |
| `XDNA_PROFILE_MAPPED_CPU` | 0 | per-handler counters (`mapped_cpu_stats()`) | cpp:1535 | debug | used by tests, tuner tools | KEEP | kept |
| `XDNA_PROFILE_FALLBACK` | 0 | per-op CPU-fallback counters | cpp:5736; training_coverage.py:13 | debug | coverage tool | KEEP | kept |
| `XDNA_ZERO_COPY_FALLBACK` | 1 | `0` uses PyTorch's copying CPU fallback | cpp:5895 | debug | bisecting tool for fallback bugs | KEEP-BUT-DOCUMENT | kept |
| `XDNA_ADOPT_HOST_OUTPUTS` | 1 | `0` skips the CPU allocator wrapper | cpp:5933 | debug | README | KEEP | kept |
| `TORCH_XDNA2_CACHE`, `TORCH_XDNA2_BUILD_VERBOSE`, `XRT_INCLUDE_DIR`, `XRT_LIB_DIR` | -- | build/cache locations | `_build.py` | config | README | KEEP | kept |
| `XDNA_TRAIN_PHYSICAL` | unset | gate for the pytest hardware tests | tests | test | -- | KEEP | kept |

## 3. Other mechanisms

| Mechanism | Verdict | Note |
|---|---|---|
| `npu_mode` 2/1/0 (`set_npu_mode`, `g_npu_mode`) | KEEP | one switch for the whole NPU surface; the tuner picks it |
| Deferred NPU dW (`set_pending_host_write`) | KEEP | core; hides the 2.6 ms context switch |
| `AsyncHostWorkQueue` (cpp:564-666) | REMOVE | only used by `XDNA_DEFER_CPU_DW`; its `drain()` in `synchronize` goes with it |
| Virtual values (`XdnaVirtualValue`, `materialize_virtual`, lazy storage growth) | REMOVE | only produced when `XDNA_LAZY_DECODER_VALUES=1` |
| Unused C pack entry points in `conv_pack.cpp` (`dw_a_bf16`, `dw_a_bhw_bf16`, `dy_bhw_bf16`, `dy_bf16`, `weight_dx_bf16`, `yxbc_slice_to_channels_last_bf16`) | REMOVE | never called |
| Python exports `fullframe_cat2_upsample2_forward`, `configure_fullframe_dw584`, `configure_conv_family`, `*_configured` of removed families | REMOVE | go with their families |

## 4. Evidence and method

* Reachability: temporary counters on every conv dispatch decision, run with
  `XDNA_NPU_MODE=2 XDNA_PROFILE_MAPPED_CPU=1` on style-transfer (`style_phases.py`, N=20), and on
  `style_labs.py` `seg_patch`, `style`, `style7`, `seg_full`,
  `seg_infer`, `style_infer`. conv3x3w took every B40 and every B1 inference
  conv that is NPU-eligible; the patch family took none. The full-frame workload used
  `fast_full` (res146 fwd), `full` (fwd292/584), `fast_dx584` (all dX of 146/292/584) and
  `fast_conv11`. VGG convs, 3->64, 64->1 and the 9x9 resnet convs stay on the CPU (cost model).
* Toggle A/B on `seg_full` (B4 584x564, `XDNA_NPU_MODE=2`), ABBA over nine
  configurations (18 runs). The machine switched from battery to AC power during the
  series and run-to-run drift was ~14 % (default: 3660 ms first, 4159 ms last), so only
  gaps larger than that are meaningful: fast full-frame off +7 %, conv11 off +7 %,
  legacy full-frame off +7 %, CPU dW chunking off +4.5 %, CPU dW overlap off +30 %,
  deferred CPU dW +50 %, lazy values +2 % (noise), lazy values + lazy dW +8 %.
* Patch conv scenarios (`style_phases.py`, N=40, `XDNA_NPU_MODE=2`, AC, ABBA): see the
  family table.
* GEMM micro-benchmark: `mm` of the three shapes, NPU vs CPU bf16, 60 samples each.
* History: `git log -S<name>`: all removed experiments except `HOLD_CPU_DW` come from the
  initial commit (`80cb980`); the full-frame dW artifact was never committed.

## 5. Counts at 9e1a5de

Rows in sections 1-3 by verdict (a combined row counts once per variable it names in
the summary below): see the final report in the branch description.
