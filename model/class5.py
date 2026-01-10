import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    Stochastic Depth (per-sample).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with tensors of shape (B, C, D, H, W)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation for 3D feature maps.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        s = F.adaptive_avg_pool3d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConvBNAct3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualSEBlock3D(nn.Module):
    """
    Residual block:
      - depthwise-ish 3D conv (optional)
      - pointwise conv
      - SE attention
      - droppath
    """

    def __init__(self, channels: int, drop_path: float = 0.0, use_dw: bool = True):
        super().__init__()
        if use_dw:
            # depthwise conv
            self.conv1 = ConvBNAct3D(channels, channels, k=3, s=1, p=1, g=channels)
            self.conv2 = ConvBNAct3D(channels, channels, k=1, s=1, p=0, g=1)
        else:
            self.conv1 = ConvBNAct3D(channels, channels, k=3, s=1, p=1, g=1)
            self.conv2 = ConvBNAct3D(channels, channels, k=3, s=1, p=1, g=1)

        self.se = SEBlock3D(channels, reduction=8)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.drop_path(x)
        return x + identity


class FPNFusion3D(nn.Module):
    """
    Lightweight multi-scale fusion:
    - input: [c2, c3, c4] (low->high resolution)
    - output: fused feature at c3 resolution (or configurable)
    """

    def __init__(self, c2, c3, c4, out_ch):
        super().__init__()
        self.l2 = nn.Conv3d(c2, out_ch, 1, bias=False)
        self.l3 = nn.Conv3d(c3, out_ch, 1, bias=False)
        self.l4 = nn.Conv3d(c4, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.smooth = ConvBNAct3D(out_ch, out_ch, k=3, s=1, p=1)

    def forward(self, f2, f3, f4):
        # project to same channels
        p2 = self.l2(f2)
        p3 = self.l3(f3)
        p4 = self.l4(f4)

        # upsample deeper features to f3 size
        target = f3.shape[-3:]  # (D,H,W)
        p4u = F.interpolate(p4, size=target, mode="trilinear", align_corners=False)

        # downsample shallower features to f3 size
        # (f2 is higher resolution; bring it to f3 by interpolate)
        p2d = F.interpolate(p2, size=target, mode="trilinear", align_corners=False)

        fused = p2d + p3 + p4u
        fused = self.bn(fused)
        fused = F.silu(fused, inplace=True)
        fused = self.smooth(fused)
        return fused


class BrainCT3DClassifier(nn.Module):
    """
    Input:  (B, 1, 96, 256, 256)
    Output: (B, num_classes) logits
    """

    def __init__(
            self,
            num_classes: int = 5,
            base_channels: int = 32,
            drop_path_rate: float = 0.10,
            dropout: float = 0.20,
    ):
        super().__init__()

        # Stem: reduce H,W early (CT is large 256x256)
        self.stem = nn.Sequential(
            ConvBNAct3D(1, base_channels, k=3, s=(1, 2, 2), p=1),  # -> (96,128,128)
            ConvBNAct3D(base_channels, base_channels, k=3, s=1, p=1),
        )

        # Stage 2
        c2 = base_channels * 2
        self.down2 = ConvBNAct3D(base_channels, c2, k=3, s=(2, 2, 2), p=1)  # -> (48,64,64)
        self.block2 = nn.Sequential(
            ResidualSEBlock3D(c2, drop_path=drop_path_rate * 0.3),
            ResidualSEBlock3D(c2, drop_path=drop_path_rate * 0.3),
        )

        # Stage 3
        c3 = base_channels * 4
        self.down3 = ConvBNAct3D(c2, c3, k=3, s=(2, 2, 2), p=1)  # -> (24,32,32)
        self.block3 = nn.Sequential(
            ResidualSEBlock3D(c3, drop_path=drop_path_rate * 0.6),
            ResidualSEBlock3D(c3, drop_path=drop_path_rate * 0.6),
            ResidualSEBlock3D(c3, drop_path=drop_path_rate * 0.6),
        )

        # Stage 4
        c4 = base_channels * 8
        self.down4 = ConvBNAct3D(c3, c4, k=3, s=(2, 2, 2), p=1)  # -> (12,16,16)
        self.block4 = nn.Sequential(
            ResidualSEBlock3D(c4, drop_path=drop_path_rate),
            ResidualSEBlock3D(c4, drop_path=drop_path_rate),
            ResidualSEBlock3D(c4, drop_path=drop_path_rate),
        )

        # Multi-scale fusion (FPN-lite)
        fpn_ch = base_channels * 4
        self.fpn = FPNFusion3D(c2=c2, c3=c3, c4=c4, out_ch=fpn_ch)

        # Head: global pooling + classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.LayerNorm(fpn_ch),
            nn.Dropout(p=dropout),
            nn.Linear(fpn_ch, num_classes),
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 2
        f2 = self.down2(x)
        f2 = self.block2(f2)

        # Stage 3
        f3 = self.down3(f2)
        f3 = self.block3(f3)

        # Stage 4
        f4 = self.down4(f3)
        f4 = self.block4(f4)

        # Fuse multi-scale features
        fused = self.fpn(f2, f3, f4)

        # Class logits
        logits = self.head(fused)
        return logits


def create_model(config=None):
    """创建增强的模型实例"""
    if config is None:
        config = {
            'base_channels': 64,  # 增加基础通道数
            'num_classes': 5,  # 六分类
        }

        model = BrainCT3DClassifier(
            base_channels=config['base_channels'],
            num_classes=config['num_classes']
        )
    model = BrainCT3DClassifier(
        base_channels=config.base_channels,
        num_classes=config.num_classes
    )

    return model


if __name__ == "__main__":
    m = BrainCT3DClassifier(num_classes=5).cuda()
    x = torch.randn(2, 1, 96, 256, 256).cuda()
    y = m(x)
    print(y.shape)