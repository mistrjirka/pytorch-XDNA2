"""A small image-to-image network and perceptual loss used by the benchmarks.

``StyleNet`` is a typical fast style-transfer / restoration generator: a strided
convolutional encoder, a stack of residual blocks, and a decoder that upsamples with
nearest-neighbour resize + convolution and concatenates skip connections.
``StyleLoss`` is L1 plus a VGG19 perceptual loss on three early feature maps.
Together with batch 40 and 36x36 patches they form the "style-transfer step" that the
documentation refers to.

No data or weights are downloaded unless asked for: the VGG19 weights are random
unless ``STYLE_NET_PRETRAINED=1`` is set (timing does not depend on them), and
``SyntheticPatches`` generates input/target pairs.  ``ImagePatches`` reads two folders
of paired images for real-data runs.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import vgg19


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.relu(y + x)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.norm(self.conv(self.upsample(x))))


class StyleNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_blocks: int = 5):
        super().__init__()
        self.conv0 = ConvBlock(in_channels, 64, stride=1)
        self.conv1 = ConvBlock(64, 128, stride=2)
        self.conv2 = ConvBlock(128, 128, stride=2)
        self.resnet_blocks = nn.ModuleList([ResBlock(128) for _ in range(num_blocks)])
        self.upconv2 = UpsampleConvBlock(128 + 128, 128)
        self.upconv1 = UpsampleConvBlock(128 + 128, 128)
        self.conv11 = ConvBlock(128 + 64 + in_channels, 64, stride=1)
        self.conv12 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        r = x2.clone()
        for block in self.resnet_blocks:
            r = block(r)
        y = self.upconv2(torch.cat([r, x2], dim=1))
        y = self.upconv1(torch.cat([y, x1], dim=1))
        y = self.conv11(torch.cat([y, x0, x], dim=1))
        return self.conv12(y)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, x):
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


class PerceptualVGG19(nn.Module):
    """Frozen VGG19 features; the outputs of ``feature_layers`` are flattened and concatenated."""

    def __init__(self, feature_layers=(0, 3, 5), use_norm=True, device="cpu"):
        super().__init__()
        weights = "DEFAULT" if os.environ.get("STYLE_NET_PRETRAINED") == "1" else None
        model = vgg19(weights=weights).eval()
        dtype = torch.bfloat16 if torch.device(device).type == "xdna" else torch.float32
        self.model = model.to(device=device, dtype=dtype)
        self.feature_layers = tuple(feature_layers)
        self.use_norm = use_norm
        for p in self.parameters():
            p.requires_grad = False
        self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x):
        y = self.norm(x) if self.use_norm else x
        maps = []
        for i in range(max(self.feature_layers) + 1):
            y = self.model.features[i](y)
            if i in self.feature_layers:
                maps.append(y.clone().view(y.shape[0], -1))
        return torch.cat(maps, dim=1)


class StyleLoss(nn.Module):
    """``l_perceptual`` * mean squared VGG feature difference + ``l_l1`` * L1."""

    def __init__(self, l_l1: float = 4.0, l_perceptual: float = 6.0, device: str = "cpu"):
        super().__init__()
        self.perceptual_vgg = PerceptualVGG19((0, 3, 5), True, device)
        self.l1_loss = nn.L1Loss()
        self.l_l1 = l_l1
        self.l_perceptual = l_perceptual

    def compute_perceptual_loss(self, input, target):
        input_features = self.perceptual_vgg(input)
        with torch.no_grad():
            target_features = self.perceptual_vgg(target)
        return ((input_features - target_features) ** 2).mean()

    def forward(self, input, target):
        return (self.l_perceptual * self.compute_perceptual_loss(input, target)
                + self.l_l1 * self.l1_loss(input, target))


class SyntheticPatches(Dataset):
    """Deterministic random image pairs: smooth noise as input, a colour-graded copy as target."""

    def __init__(self, patch_size=36, num_patches=200000, seed=0):
        self.patch_size, self.num_patches, self.seed = patch_size, num_patches, seed

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        s = self.patch_size
        low = torch.rand(3, s // 4 + 2, s // 4 + 2, generator=g)
        x = F.interpolate(low[None], size=(s, s), mode="bilinear", align_corners=False)[0]
        gain = torch.tensor([1.2, 0.9, 1.1]).view(3, 1, 1)
        y = (x * gain + 0.05).clamp(0, 1) ** 0.9
        return x * 2 - 1, y


class ImagePatches(Dataset):
    """Random aligned crops from paired image folders (same file names in both)."""

    def __init__(self, input_dir, target_dir, patch_size=36, num_patches=200000):
        import torchvision.io as tio
        names = sorted(n for n in os.listdir(input_dir) if os.path.exists(os.path.join(target_dir, n)))
        if not names:
            raise ValueError("no paired images found")
        read = lambda d, n: tio.read_image(os.path.join(d, n), tio.ImageReadMode.RGB).float() / 255.0
        self.inputs = torch.stack([read(input_dir, n) for n in names])
        self.targets = torch.stack([read(target_dir, n) for n in names])
        mean, std = self.inputs.mean(dim=(0, 2, 3)), self.inputs.std(dim=(0, 2, 3))
        self.inputs = (self.inputs - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        self.patch_size, self.num_patches = patch_size, num_patches

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        i = torch.randint(0, self.inputs.shape[0], (1,)).item()
        _, h, w = self.inputs.shape[1:]
        y = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        x = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        s = self.patch_size
        return self.inputs[i, :, y:y + s, x:x + s], self.targets[i, :, y:y + s, x:x + s]


def make_dataset(patch_size=36, num_patches=200000, input_dir=None, target_dir=None):
    if input_dir and target_dir:
        return ImagePatches(input_dir, target_dir, patch_size, num_patches)
    return SyntheticPatches(patch_size, num_patches)
