# Performance

[Back to the README](../README.md)

## Recorded model-zoo results

These measurements were recorded on 2026-10-13 for backend commit `fae72c2`.
The table was checked against `benchmarks/results/zoo_fae72c2_ac.json` during
review of the maintainer's local documentation cleanup `78f7531`. That raw run
file has not been published in this repository; the reviewed summary is below.

Setup: Ryzen AI 9 365, XRT 2.21.75, PyTorch 2.14.1, Python 3.12, BF16,
AC power and 16 CPU threads. The harness uses synthetic inputs, AdamW, 24 warmup
steps and 10 timed steps. CPU and XDNA run in separate processes in alternating
order, CPU/XDNA then XDNA/CPU. Each reported time is the median of the two
per-round medians and includes forward, loss, backward and the optimizer step.

| Model | Input (batch) | CPU ms | XDNA ms | Speedup |
|---|---|---|---|---|
| resnet18 | 3x64x64 (32) | 53.7 | 46.6 | 1.15x |
| resnet18 | 3x224x224 (8) | 98.2 | 84.7 | 1.16x |
| vgg11 | 3x128x128 (8) | 225.0 | 173.3 | 1.30x |
| mobilenet_v3_small | 3x96x96 (32) | 55.8 | 57.4 | 0.97x |
| 3-level U-Net | 3x96x96 (8) | 69.1 | 48.4 | 1.43x |
| Transformer encoder, 4 layers, d=256 | 128 tokens (16) | 87.0 | 83.4 | 1.04x |
| MLP 784-1024-1024-10 | 784 (256) | 3.8 | 3.3 | 1.16x |
| LSTM, 2 layers, 256 hidden | 64x64 (32) | 18.6 | 18.0 | 1.03x |
| 1-D conv audio classifier | 16000 samples (16) | 26.1 | 24.6 | 1.06x |
| Conv autoencoder | 3x128x128 (16) | 27.3 | 27.2 | 1.01x |
| Embedding + MLP | 64 tokens (128) | 5.5 | 2.7 | 2.03x |
| NIS style-transfer net (MyNet) | 3x36x36 (40) | 145.3 | 93.0 | 1.56x |

The geometric mean of CPU-time / XDNA-time over all 12 workloads is **1.21x**.
Ratios above 1 favor XDNA. The MobileNet result is slower in this measurement;
a few percent difference in a short laptop run does not establish a reliable
advantage for either backend. Power state, temperature and background load can
change the result.

The MyNet row uses synthetic data, L1 loss and AdamW. It is separate from the real
style-transfer application with VGG perceptual loss below. These short training
runs check loss agreement; they do not establish final model quality or accuracy.

## What is accelerated

General NPU convolution supports BF16 3x3 filters with stride 1, padding 1,
dilation 1 and one convolution group. Shape and cost checks can retain a CPU path.
Images wider than 36 columns are processed as strips. There are also bundled
full-frame decoder programs for the DRIVE workload and three fixed matrix
multiply shapes `(M, K, N)`:

```text
(256, 768, 3072)
(256, 3072, 768)
(3072, 256, 768)
```

Other matrix shapes, attention, recurrent layers, depthwise and 1x1 convolutions,
normalization and elementwise operations use CPU paths. Speedups on these workloads
can come from host-side changes: fewer copies, fused CPU recurrent kernels,
fused optimizers and backward-thread scheduling. A faster XDNA training step does
not imply that every operation ran on the NPU.

General NPU convolutions use BF16 activations with BFP16 weights/operands. Previous
numerical tests measured roughly 1% relative error against FP32, compared with
roughly 0.2% for CPU BF16. A 300-step NIS comparison is recorded by
[`bfp16_training_gate.py`](../benchmarks/bfp16_training_gate.py) and its
[results](../benchmarks/results/bfp16_gate.log). Validate numerical behavior for
your own model and loss.

## Application-specific measurements

The previous README recorded these separately from the model-zoo run:

| Application | CPU BF16 | XDNA BF16 | Measurement |
|---|---:|---:|---|
| NIS style transfer, 2026-10-10 | 171.8 ms/step | 128.8 ms/step | Training step with VGG loss and Adam |
| NIS style transfer, 2026-10-13 | 187.7 ms/step | 135.5 ms/step | One comparison round, VGG loss and Adam |
| DRIVE segmentation, earlier full-frame checkpoint | 0.151 to 0.155 it/s | 0.210 to 0.212 it/s | Training plus validation |

The style-transfer runs use batch 40 and 36x36 patches. The
[NIS harness](../benchmarks/nis_style_transfer.py) depends on the maintainer's
coursework scripts, Python environment and data outside this repository; it cannot
be reproduced from a fresh clone alone.

The DRIVE measurement uses `MyNet(3,1,7)`, batch 4 and 584x564 images. It predates
the 2026-10-12 cleanup and has not been rerun since that cleanup. For historical
optimization notes, see the [optimization log](OPTIMIZATION_LOG.md).

## Reproducing the measurements

Run from a checkout in the configured Python/XRT/IRON environment. Install a
`torchvision` build compatible with your PyTorch build for ResNet, VGG and MobileNet.
For example, compare three of the self-contained workloads:

```bash
PYTHONPATH=. python benchmarks/model_zoo.py \
  --models resnet18,unet,mobilenet_v3 \
  --steps 10 --warmup 24 --rounds 2 --threads 16 \
  --json /tmp/torch-xdna2-model-zoo.json
```

Use the keys in [model_zoo.py](../benchmarks/model_zoo.py) to select other models.
The default list includes `nis_mynet`, which imports `MyNet` from coursework
outside the repository; omit it on a fresh clone. Reproducing all 12 rows requires
that external model source. The real application benchmarks also require external
data.

Use AC power and record the power profile, hardware/software versions, thread
counts, input shapes and environment overrides. Run hardware benchmarks serially.
Allow kernel builds to finish and the tuner to choose a mode before collecting
steady-state timings; cold-start time is a separate measurement.

Include `optimizer.step()` or explicit reads of the gradients. Calling
`torch.xdna.synchronize()` after backward alone omits deferred weight-gradient
computation. For diagnostics, [usage](usage.md#check-which-paths-a-workload-uses)
explains the NPU dispatch counters; disable profiling for final timings.

Additional tools in [benchmarks](../benchmarks/) cover convolution shapes,
operator coverage and BFP16 training behavior. [Development](development.md)
lists the correctness checks to run alongside performance tests.
