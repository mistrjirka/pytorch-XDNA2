#!/bin/bash
# Compile-only build of the upstream mlir-aie whole-array GEMM (bf16 in, BFP16-emulated multiply).
# usage: build_gemm.sh OUTDIR M K N m k n [cols=8] [dout=bf16|f32]
# Needs an mlir-aie checkout (MLIR_AIE_DIR) and its IRON Python (IRON_PYTHON, e.g. the managed
# environment from `torch-xdna2-setup`).  Builds need no NPU; run the result with run_gemm.py.
set -e
OUT=$1; M=$2; K=$3; N=$4; m=$5; k=$6; n=$7; COLS=${8:-8}; DOUT=${9:-bf16}
: "${MLIR_AIE_DIR:?set MLIR_AIE_DIR to an mlir-aie checkout}"
: "${IRON_PYTHON:?set IRON_PYTHON to the IRON environment python}"
mkdir -p "$OUT"
tag=bf16_${DOUT}out_${M}x${K}x${N}_${m}x${k}x${n}_${COLS}c
cd "$MLIR_AIE_DIR/programming_examples/basic/matrix_multiplication/whole_array"
PATH="$(dirname "$IRON_PYTHON"):$PATH" python3 whole_array.py --dev npu2 -M $M -K $K -N $N -m $m -k $k -n $n \
  --n-aie-cols $COLS --dtype_in bf16 --dtype_out $DOUT --emulate-bf16-mmul-with-bfp16 1 \
  --xclbin-path "$OUT/$tag.xclbin" --insts-path "$OUT/$tag.bin"
echo "$OUT/$tag.xclbin $OUT/$tag.bin"
