"""NIS-specific declarations for the generic regions (kept out of the library).

`fuse_nis_norm_act(model)` pairs each BatchNorm2d with the activation that is
applied directly to its output in the coursework MyNet:

  ConvBlock / UpsampleConvBlock:  conv -> norm -> relu                 (pair norm, relu)
  ResNetBlock:  norm1 -> relu is direct (pair norm1, relu); norm2 feeds an add
                first (`relu(norm2(..) + x_in)`), so norm2 is NOT paired.  The
                shared relu module is safe: it passes through only tensors the
                fused norm1 produced and activates everything else.
"""
import torch.nn as nn

from xdna_train import regions


def fuse_nis_norm_act(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        kind = type(m).__name__
        if kind in ("ConvBlock", "UpsampleConvBlock"):
            n += bool(regions.fuse_norm_activation(m.norm, m.relu))
        elif kind == "ResNetBlock":
            n += bool(regions.fuse_norm_activation(m.norm1, m.relu))
    return n
