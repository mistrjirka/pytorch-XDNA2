# Third-party notices

`pytorch-XDNA2` is licensed under the MIT License. The project depends on and/or
uses generated artifacts from third-party projects with their own licenses.

## PyTorch

PyTorch is an external Python dependency and is not redistributed by this
repository. PyTorch's main project license is BSD-3-Clause. See:
https://github.com/pytorch/pytorch/blob/main/LICENSE

## AMD/Xilinx XRT

XRT is an external system/runtime dependency and is dynamically linked at
build/runtime; XRT itself is not vendored in this package. XRT source files are
licensed under Apache-2.0. See:
https://github.com/Xilinx/XRT

The code in `xdna_train/_vendor/xrt_shim/` is this project's adapter layer
over the public XRT C++ API; it is not a copy of the XRT implementation.

## MLIR-AIE / AIE2P matrix kernel

The bundled XDNA instruction/program artifacts were built using the
MLIR-AIE AIE2P matrix kernel, including `aie_kernels/aie2p/mm.cc`, which is
Copyright (C) 2025 Advanced Micro Devices, Inc. and licensed under
Apache-2.0 WITH LLVM-exception.

A copy of the upstream MLIR-AIE license is included at
`LICENSES/MLIR-AIE.txt`.

Source:
https://github.com/Xilinx/mlir-aie
