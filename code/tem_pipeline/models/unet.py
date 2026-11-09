"""The notebook's two-input SmallUNet, with unchanged state_dict key names."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=False),
        )


class SmallUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()

        if base <= 0 or base % 4:
            raise ValueError("base_channels must be a positive multiple of 4 for GroupNorm")

        widths = [base * 2**index for index in range(5)]
        self.encoders = nn.ModuleList(
            [
                ConvBlock(2 if index == 0 else widths[index - 1], width)
                for index, width in enumerate(widths)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ConvBlock(widths[index + 1] + widths[index], widths[index])
                for index in range(3, -1, -1)
            ]
        )
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []

        for index, encoder in enumerate(self.encoders):
            if index:
                x = F.max_pool2d(x, 2)

            x = encoder(x)

            if index < 4:
                skips.append(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))

        return self.head(x)
