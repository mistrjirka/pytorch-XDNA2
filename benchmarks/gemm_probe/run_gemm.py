"""Run an IRON matmul xclbin (bf16 in, bf16 or f32 out) on the NPU and time it.

System Python with pyxrt and numpy (python3.14 on Arch):
    python3.14 run_gemm.py X.xclbin X.bin M K N [iters] [f32]
"""
import sys, time
import numpy as np
import pyxrt

xclbin, insts, M, K, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
iters = int(sys.argv[6]) if len(sys.argv) > 6 else 20
out_f32 = len(sys.argv) > 7 and sys.argv[7] == 'f32'


def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def bf16_val(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


rng = np.random.default_rng(0)
A = bf16_bits(rng.standard_normal((M, K)).astype(np.float32) * 0.1)
B = bf16_bits(rng.standard_normal((K, N)).astype(np.float32) * 0.1)
ref = bf16_val(A) @ bf16_val(B)

dev = pyxrt.device(0)
xb = pyxrt.xclbin(xclbin)
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
name = xb.get_kernels()[0].get_name()
k = pyxrt.kernel(ctx, name)
instr = np.fromfile(insts, dtype=np.uint32)
bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
bo_a = pyxrt.bo(dev, A.nbytes, pyxrt.bo.host_only, k.group_id(3))
bo_b = pyxrt.bo(dev, B.nbytes, pyxrt.bo.host_only, k.group_id(4))
bo_c = pyxrt.bo(dev, M * N * (4 if out_f32 else 2), pyxrt.bo.host_only, k.group_id(5))
bo_t = pyxrt.bo(dev, 1, pyxrt.bo.host_only, k.group_id(6))
bo_r = pyxrt.bo(dev, 4, pyxrt.bo.host_only, k.group_id(7))
bo_i.write(instr.tobytes(), 0); bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_a.write(A.tobytes(), 0); bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_b.write(B.tobytes(), 0); bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
times = []
for i in range(iters + 3):
    t = time.perf_counter()
    r = k(3, bo_i, len(instr), bo_a, bo_b, bo_c, bo_t, bo_r)
    r.wait()
    times.append((time.perf_counter() - t) * 1e3)
bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
C = (np.frombuffer(bo_c.read(M * N * 4, 0), dtype=np.float32).reshape(M, N) if out_f32 else bf16_val(np.frombuffer(bo_c.read(M * N * 2, 0), dtype=np.uint16).reshape(M, N)))
times = sorted(times[3:])
rel = float(np.linalg.norm(C - ref) / np.linalg.norm(ref))
flops = 2.0 * M * K * N
print(f"{M}x{K}x{N}: min {times[0]:.3f} ms  median {times[len(times)//2]:.3f} ms  "
      f"=> {flops / times[0] / 1e9:.2f} / {flops / times[len(times)//2] / 1e9:.2f} TFLOPS (min/median), rel err vs fp32 {rel:.4f}")
