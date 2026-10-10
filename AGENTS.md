# AGENTS.md — torch-xdna2

**Read [docs/AGENT_HANDOFF_2026-10-10.md](docs/AGENT_HANDOFF_2026-10-10.md) first.** It records the current working source, remote-machine/Python build setup, real-data benchmark commands, accepted/rejected experiments, physical correctness results, relevant Pi skill locations and the next optimization work.

## Current project goal
Make a general-purpose **PyTorch XDNA2 training backend** faster than native CPU BF16 when NPU offloading is profitable. Do not silently slow down CPU-only paths or claim speedup based only on NPU kernel time. The real NIS style-transfer benchmark is an admission/regression test, **not** a license to hardcode that specific model.

## Key locations
- **Maintained source**: this repository, particularly xdna_train/_native/torch_xdna_device.cpp and xdna_train/device.py.
- **Experiment and benchmark workspace**: /home/jirka/programy/torch-xdna-train; real script benchmark in scripts/benchmark_original_style.py.
- **Real application**: /home/jirka/programovani/magistr/nis/lab02_cnn/style_transfer_test.py.
- **Pi skills**: /home/jirka/.pi/agent/skills (symlinks to the wiki-generated SKILL.md files); particularly diagnosing-bug, running-real-data-smoke-tests, review-final-diff, impl-check and investigate-feasibility.
- **Python**: /home/jirka/programovani/magistr/nis/.venv/bin/python; its installed xdna_train may be older than this checkout. Set PYTHONPATH to this repository and verify xdna_train.__file__ and native_library_path().

## Verified 2026-10-10
- With real training patches (B40, 36×36, VGG loss), CPU BF16 ~233 ms/step (4.29 it/s); optimized XDNA BF16 ~519 ms/step (1.93 it/s).
- XDNA_CONV_OUTPUT_NCHW (small-patch Conv output as NCHW) improved XDNA training ~15% on paired runs and is now the default; set 0 for the raw layout.
- Later on 2026-10-10: the conv3x3w forward/dX/dW families (README) brought the NIS script to 147.7 ms/step vs CPU BF16 171.4 (1.16x).
- XDNA_CONV_PACK_THREADS=4 and XDNA_BO_CACHE_MB=1024 were the tested settings, not universal defaults.
- Physical Conv→BatchNorm output/gradient tests passed on H18/H36. Wider kernel coverage and backward/fallback scheduling remain unsolved.

**Do not erase or reset existing uncommitted changes, modify the NIS coursework to massage timings, or edit site-packages during experimentation.** Keep optimized behavior opt-in until correctness and representative end-to-end performance are proven. See the full handoff for commands and detailed evidence.
