"""
Standard U-Net for binary segmentation.
Input: 3xHxW normalized image. Output: 1xHxW logits (apply sigmoid for prob).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if shapes differ by 1px due to odd dims
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        """
        base=32 keeps the model small enough to train on CPU in a reasonable time.
        Standard U-Net papers use base=64.
        """
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16, base * 8)
        self.u2 = Up(base * 8, base * 4)
        self.u3 = Up(base * 4, base * 2)
        self.u4 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(model_type="scratch", base_channels=16, encoder_name="resnet34",
               encoder_weights="imagenet", in_channels=3, out_channels=1):
    """
    model_type:
        "scratch"     -> our small from-scratch U-Net (base_channels controls size)
        "pretrained"  -> segmentation-models-pytorch U-Net with an ImageNet-
                         pretrained encoder (encoder_name + encoder_weights)
    """
    if model_type == "scratch":
        return UNet(in_channels=in_channels, out_channels=out_channels,
                    base=base_channels)
    if model_type == "pretrained":
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )
    raise ValueError(f"unknown model_type: {model_type}")
