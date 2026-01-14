import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------
# Basic Blocks (2D)
# -------------------------
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class MultiScaleFeatureExtractor2D(nn.Module):
    """Multi-scale feature extraction (2D)"""

    def __init__(self, in_channels, base_channels):
        super().__init__()
        # Keep the same channel split strategy
        self.branch_channels = [base_channels // 4, base_channels // 2, base_channels // 4, base_channels // 4]
        self.total_channels = sum(self.branch_channels)

        self.conv1x1 = nn.Conv2d(in_channels, self.branch_channels[0], kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(self.branch_channels[0])

        self.conv3x3 = ResidualBlock2D(in_channels, self.branch_channels[1])

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, self.branch_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.branch_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch_channels[2], self.branch_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.branch_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.dilated_conv = nn.Conv2d(in_channels, self.branch_channels[3],
                                      kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn_dilated = nn.BatchNorm2d(self.branch_channels[3])

        self.fusion = nn.Conv2d(self.total_channels, base_channels, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(base_channels)

    def forward(self, x):
        scale1 = F.relu(self.bn1x1(self.conv1x1(x)))
        scale2 = self.conv3x3(x)
        scale3 = self.conv5x5(x)
        scale4 = F.relu(self.bn_dilated(self.dilated_conv(x)))

        out = torch.cat([scale1, scale2, scale3, scale4], dim=1)
        out = self.bn_fusion(self.fusion(out))
        return F.relu(out)


# -------------------------
# CBAM (2D)
# -------------------------
class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


class ChannelAttention2D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * att


class CBAM2D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention2D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention2D()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# -------------------------
# Positional Encoding (2D) - vectorized (no triple loops)
# -------------------------
class PositionalEncoding2D(nn.Module):
    """
    Sin-cos positional encoding for 2D feature maps:
      x: (B, C, H, W)  ->  x + pe (1, C, H, W)
    """
    def __init__(self, channels, max_dim=128):
        super().__init__()
        self.channels = channels
        self.max_dim = max_dim
        self.register_buffer("pos_encoding", torch.empty(0), persistent=False)

    def forward(self, x):
        b, c, h, w = x.shape
        if self.pos_encoding.numel() == 0 or self.pos_encoding.shape[-2:] != (h, w) or self.pos_encoding.shape[1] != c:
            device = x.device
            pe = torch.zeros((1, c, h, w), device=device)

            # normalized coords
            yy = torch.arange(h, device=device).float() / float(self.max_dim)
            xx = torch.arange(w, device=device).float() / float(self.max_dim)
            yv, xv = torch.meshgrid(yy, xx, indexing="ij")  # (H, W)

            # allocate channels: groups of 4 (sin_y, cos_y, sin_x, cos_x)
            # if c not divisible by 4, last channels are filled progressively
            for i in range(0, c, 4):
                div = 10000.0 ** (-(i / max(c, 1)))
                if i < c:
                    pe[0, i, :, :] = torch.sin(yv * div)
                if i + 1 < c:
                    pe[0, i + 1, :, :] = torch.cos(yv * div)
                if i + 2 < c:
                    pe[0, i + 2, :, :] = torch.sin(xv * div)
                if i + 3 < c:
                    pe[0, i + 3, :, :] = torch.cos(xv * div)

            self.pos_encoding = pe

        return x + self.pos_encoding


# -------------------------
# Transformer Block (sequence)
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, N, C)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        return x


# -------------------------
# Fusion: MultiHead only (2D)
# -------------------------
class MultiHeadAttentionFusion(nn.Module):
    """Multi-head attention fusion for global (vector) and local (feature map)"""

    def __init__(self, global_dim, local_dim, fusion_dim, num_heads=8):
        super().__init__()
        self.to_q = nn.Linear(global_dim, fusion_dim)
        self.to_kv = nn.Linear(local_dim, fusion_dim * 2)

        self.multihead_attn = nn.MultiheadAttention(
            fusion_dim, num_heads, batch_first=True, dropout=0.1
        )

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(0.1),
        )

        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        self.residual_proj = nn.Linear(global_dim, fusion_dim)
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, global_feat, local_feat):
        """
        global_feat: (B, global_dim)
        local_feat:  (B, local_dim, H, W)
        """
        b = global_feat.size(0)

        # 1) pool local
        local_pooled = F.adaptive_avg_pool2d(local_feat, 1).view(b, -1)  # (B, local_dim)

        # 2) project
        q = self.to_q(global_feat).unsqueeze(1)  # (B, 1, fusion_dim)
        kv = self.to_kv(local_pooled)            # (B, 2*fusion_dim)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1)  # (B, 1, fusion_dim)
        v = v.unsqueeze(1)  # (B, 1, fusion_dim)

        # 3) attention
        attn_out, _ = self.multihead_attn(q, k, v)  # (B, 1, fusion_dim)
        attn_out = attn_out.squeeze(1)              # (B, fusion_dim)

        # 4) residual + norm
        global_proj = self.residual_proj(global_feat)  # (B, fusion_dim)
        x = self.norm1(attn_out + global_proj)

        # 5) FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)

        # 6) output proj
        return self.output_proj(x)


# -------------------------
# Main Model (2D)
# -------------------------
class Medical2DClassifierWithMultiHeadFusion(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_classes=2, input_shape=(256, 256)):
        super().__init__()
        self.input_shape = input_shape

        # initial conv
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(base_channels)

        # multi-scale trunk
        self.multiscale1 = MultiScaleFeatureExtractor2D(base_channels, base_channels * 2)
        self.downsample1 = nn.MaxPool2d(2, stride=2)

        self.multiscale2 = MultiScaleFeatureExtractor2D(base_channels * 2, base_channels * 4)
        self.downsample2 = nn.MaxPool2d(2, stride=2)

        self.multiscale3 = MultiScaleFeatureExtractor2D(base_channels * 4, base_channels * 8)

        # CBAM
        self.cbam1 = CBAM2D(base_channels * 2)
        self.cbam2 = CBAM2D(base_channels * 4)
        self.cbam3 = CBAM2D(base_channels * 8)

        # global path (2D)
        self.global_path = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # pos encoding + transformers
        self.pos_encoding = PositionalEncoding2D(base_channels * 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(base_channels * 2, num_heads=8) for _ in range(2)
        ])

        # fusion (multihead only)
        fusion_dim = base_channels * 4
        self.fusion = MultiHeadAttentionFusion(
            global_dim=base_channels * 2,
            local_dim=base_channels * 8,
            fusion_dim=fusion_dim,
            num_heads=8
        )

        # head
        self.final_proj = nn.Linear(fusion_dim, base_channels * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(base_channels * 2),
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_channels, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # local path
        x_local = F.relu(self.initial_bn(self.initial_conv(x)))

        x_local = self.multiscale1(x_local)
        x_local = self.cbam1(x_local)
        x_local = self.downsample1(x_local)

        x_local = self.multiscale2(x_local)
        x_local = self.cbam2(x_local)
        x_local = self.downsample2(x_local)

        x_local = self.multiscale3(x_local)
        x_local = self.cbam3(x_local)  # (B, base*8, h', w')

        # global path
        x_global = self.global_path(x_local)      # (B, base*2, hg, wg)
        x_global = self.pos_encoding(x_global)    # add PE

        b, c, h, w = x_global.shape
        x_seq = rearrange(x_global, "b c h w -> b (h w) c")  # (B, N, C)

        for blk in self.transformer_blocks:
            x_seq = blk(x_seq)

        global_pooled = x_seq.mean(dim=1)  # (B, base*2)

        # fusion
        fused = self.fusion(global_pooled, x_local)  # (B, fusion_dim)

        # classify
        feat = self.final_proj(fused)
        out = self.classifier(feat)
        return out


def create_model(config=None):
    """
    config can be:
      - dict with keys: in_channels, base_channels, num_classes, input_shape
      - object/namespace with attributes above
      - None (use defaults)
    """
    if config is None:
        cfg = dict(in_channels=1, base_channels=32, num_classes=2, input_shape=(256, 256))
        return Medical2DClassifierWithMultiHeadFusion(**cfg)

    if isinstance(config, dict):
        return Medical2DClassifierWithMultiHeadFusion(
            in_channels=config.get("in_channels", 1),
            base_channels=config.get("base_channels", 32),
            num_classes=config.get("num_classes", 2),
            input_shape=config.get("input_shape", (256, 256)),
        )

    # fallback: attribute-style
    return Medical2DClassifierWithMultiHeadFusion(
        in_channels=getattr(config, "in_channels", 1),
        base_channels=getattr(config, "base_channels", 32),
        num_classes=getattr(config, "num_classes", 2),
        input_shape=getattr(config, "input_shape", (256, 256)),
    )


# -------------------------
# Example
# -------------------------
if __name__ == "__main__":
    model = create_model({
        "in_channels": 1,
        "base_channels": 32,
        "num_classes": 2,
        "input_shape": (256, 256),
    })

    test_input = torch.randn(2, 1, 256, 256)
    out = model(test_input)
    print("Output shape:", out.shape)
    print("Params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
