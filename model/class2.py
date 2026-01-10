import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


# 模型定义部分（保持与之前相同）
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取模块"""

    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.branch_channels = [base_channels // 4, base_channels // 2, base_channels // 4, base_channels // 4]
        self.total_channels = sum(self.branch_channels)

        self.conv1x1 = nn.Conv3d(in_channels, self.branch_channels[0], kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm3d(self.branch_channels[0])

        self.conv3x3 = ResidualBlock3D(in_channels, self.branch_channels[1])

        self.conv5x5 = nn.Sequential(
            nn.Conv3d(in_channels, self.branch_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.branch_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.branch_channels[2], self.branch_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.branch_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.dilated_conv = nn.Conv3d(in_channels, self.branch_channels[3],
                                      kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn_dilated = nn.BatchNorm3d(self.branch_channels[3])

        self.fusion = nn.Conv3d(self.total_channels, base_channels, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm3d(base_channels)

    def forward(self, x):
        scale1 = F.relu(self.bn1x1(self.conv1x1(x)))
        scale2 = self.conv3x3(x)
        scale3 = self.conv5x5(x)
        scale4 = F.relu(self.bn_dilated(self.dilated_conv(x)))

        out = torch.cat([scale1, scale2, scale3, scale4], dim=1)
        out = self.fusion(out)
        out = self.bn_fusion(out)
        return F.relu(out)


class SpatialAttention3D(nn.Module):
    """3D空间注意力模块"""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att_map = torch.cat([avg_out, max_out], dim=1)
        att_map = self.sigmoid(self.conv(att_map))
        return x * att_map


class ChannelAttention3D(nn.Module):
    """3D通道注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        return x * attention


class CBAM3D(nn.Module):
    """3D卷积注意力模块（结合空间和通道注意力）"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention3D()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PositionalEncoding3D(nn.Module):
    """优化的3D位置编码"""

    def __init__(self, channels, max_dim=128):
        super().__init__()
        self.channels = channels
        self.max_dim = max_dim

    def forward(self, x):
        batch, channels, depth, height, width = x.shape

        if not hasattr(self, 'pos_encoding') or self.pos_encoding.shape[2:] != (depth, height, width):
            # 预计算位置编码
            pos_enc = torch.zeros((1, channels, depth, height, width), device=x.device)

            # 为每个维度创建位置编码
            for d in range(depth):
                for h in range(height):
                    for w in range(width):
                        # 归一化位置
                        pos_d = d / self.max_dim
                        pos_h = h / self.max_dim
                        pos_w = w / self.max_dim

                        # 为每个位置生成编码
                        for i in range(0, channels, 6):
                            if i < channels:
                                pos_enc[0, i, d, h, w] = math.sin(pos_d * 10000 ** (-i / channels))
                                if i + 1 < channels:
                                    pos_enc[0, i + 1, d, h, w] = math.cos(pos_d * 10000 ** (-i / channels))
                            if i + 2 < channels:
                                pos_enc[0, i + 2, d, h, w] = math.sin(pos_h * 10000 ** (-(i + 2) / channels))
                                if i + 3 < channels:
                                    pos_enc[0, i + 3, d, h, w] = math.cos(pos_h * 10000 ** (-(i + 3) / channels))
                            if i + 4 < channels:
                                pos_enc[0, i + 4, d, h, w] = math.sin(pos_w * 10000 ** (-(i + 4) / channels))
                                if i + 5 < channels:
                                    pos_enc[0, i + 5, d, h, w] = math.cos(pos_w * 10000 ** (-(i + 5) / channels))

            self.register_buffer('pos_encoding', pos_enc, persistent=False)

        return x + self.pos_encoding


class TransformerBlock3D(nn.Module):
    """3D Transformer块"""

    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)

        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + self.dropout(mlp_output)

        return x

class CrossAttentionFusion(nn.Module):
    """基于交叉注意力的全局-局部特征融合模块"""

    def __init__(self, global_dim, local_dim, fusion_dim, num_heads=8):
        super().__init__()
        self.fusion_dim = fusion_dim

        # 全局特征投影
        self.global_proj = nn.Linear(global_dim, fusion_dim)
        self.global_norm = nn.LayerNorm(fusion_dim)

        # 局部特征投影
        self.local_proj = nn.Conv3d(local_dim, fusion_dim, kernel_size=1, bias=False)
        self.local_bn = nn.BatchNorm3d(fusion_dim)
        self.local_norm = nn.LayerNorm(fusion_dim)

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            fusion_dim, num_heads, batch_first=True, dropout=0.1
        )

        # 特征增强
        self.enhance = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )

    def forward(self, global_feat, local_feat):
        """
        Args:
            global_feat: (B, global_dim) 全局特征
            local_feat: (B, local_dim, D, H, W) 局部特征
        Returns:
            fused_feat: (B, fusion_dim) 融合后的特征
        """
        batch_size = global_feat.size(0)

        # 1. 投影到统一维度
        global_proj = self.global_proj(global_feat)  # (B, fusion_dim)
        global_proj = self.global_norm(global_proj)

        local_proj = self.local_proj(local_feat)  # (B, fusion_dim, D, H, W)
        local_proj = self.local_bn(local_proj)
        local_proj = F.relu(local_proj)

        # 2. 重塑局部特征
        local_flat = rearrange(local_proj, 'b c d h w -> b (d h w) c')  # (B, N, fusion_dim)
        local_flat = self.local_norm(local_flat)

        # 3. 交叉注意力：全局特征查询局部特征
        # 扩展全局特征作为查询
        global_query = global_proj.unsqueeze(1)  # (B, 1, fusion_dim)

        # 执行交叉注意力
        attended_local, _ = self.cross_attention(
            global_query,  # query: 全局特征
            local_flat,  # key: 局部特征
            local_flat  # value: 局部特征
        )  # (B, 1, fusion_dim)

        attended_local = attended_local.squeeze(1)  # (B, fusion_dim)

        # 4. 门控融合
        gate_input = torch.cat([global_proj, attended_local], dim=-1)
        gate_weights = self.gate(gate_input)  # (B, fusion_dim)

        # 门控加权融合
        fused = gate_weights * global_proj + (1 - gate_weights) * attended_local

        # 5. 特征增强
        fused = self.enhance(fused)

        return fused


class SpatialAwareAttentionFusion(nn.Module):
    """空间感知的注意力融合模块"""

    def __init__(self, global_dim, local_dim, fusion_dim, num_heads=8):
        super().__init__()

        # 全局特征处理
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )

        # 局部特征处理
        self.local_proj = nn.Sequential(
            nn.Conv3d(local_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm3d(fusion_dim),
            nn.GELU()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(fusion_dim, fusion_dim // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(fusion_dim // 16, fusion_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)

        # Transformer融合
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, global_feat, local_feat):
        # 1. 投影特征
        global_proj = self.global_proj(global_feat)  # (B, fusion_dim)

        local_proj = self.local_proj(local_feat)  # (B, fusion_dim, D, H, W)

        # 2. 应用空间和通道注意力到局部特征
        # 空间注意力
        avg_out = torch.mean(local_proj, dim=1, keepdim=True)
        max_out, _ = torch.max(local_proj, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_att)
        local_att = local_proj * spatial_weights

        # 通道注意力
        channel_weights = self.channel_attention(local_att)
        local_att = local_att * channel_weights

        # 3. 池化局部特征
        local_pooled = self.adaptive_pool(local_att).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, fusion_dim)

        # 4. 准备序列输入
        sequence = torch.stack([global_proj, local_pooled], dim=1)  # (B, 2, fusion_dim)

        # 5. Transformer融合
        fused_seq = self.fusion_transformer(sequence)  # (B, 2, fusion_dim)

        # 6. 聚合
        fused = torch.mean(fused_seq, dim=1)  # (B, fusion_dim)

        return fused


class MultiHeadAttentionFusion(nn.Module):
    """多头注意力特征融合"""

    def __init__(self, global_dim, local_dim, fusion_dim, num_heads=8):
        super().__init__()

        self.global_dim = global_dim
        self.local_dim = local_dim
        self.fusion_dim = fusion_dim

        # 投影层 - 修改这里
        self.to_q = nn.Linear(global_dim, fusion_dim)  # 投影到融合维度
        self.to_kv = nn.Linear(local_dim, fusion_dim * 2)  # 投影到融合维度的两倍

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            fusion_dim, num_heads, batch_first=True, dropout=0.1
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(0.1)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # 残差投影层 - 新增：将全局特征投影到融合维度
        self.residual_proj = nn.Linear(global_dim, fusion_dim)

        # 输出投影
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, global_feat, local_feat):
        batch_size = global_feat.size(0)

        # 1. 局部特征池化
        local_pooled = F.adaptive_avg_pool3d(local_feat, 1)
        local_pooled = local_pooled.view(batch_size, -1)  # (B, local_dim)

        # 2. 投影 - 修复维度问题
        q = self.to_q(global_feat).unsqueeze(1)  # (B, 1, fusion_dim)

        kv = self.to_kv(local_pooled)  # (B, fusion_dim * 2)
        k, v = kv.chunk(2, dim=-1)  # 各(B, fusion_dim)
        k = k.unsqueeze(1)  # (B, 1, fusion_dim)
        v = v.unsqueeze(1)  # (B, 1, fusion_dim)

        # 3. 多头注意力
        attn_output, attn_weights = self.multihead_attn(q, k, v)

        # 4. 残差连接和归一化 - 修复维度问题
        # 将全局特征投影到融合维度
        global_proj = self.residual_proj(global_feat)
        x = self.norm1(attn_output.squeeze(1) + global_proj)

        # 5. 前馈网络
        ffn_output = self.ffn(x)

        # 6. 残差连接和归一化
        output = self.norm2(ffn_output + x)

        # 7. 输出投影
        output = self.output_proj(output)

        return output


class Medical3DClassifierWithAttentionFusion(nn.Module):
    """使用注意力融合的3D医学图像分类模型"""

    def __init__(self, in_channels=1, base_channels=64,
                 num_classes=2, input_shape=(64, 128, 128),
                 fusion_type='cross_attention'):
        super().__init__()
        self.input_shape = input_shape
        self.fusion_type = fusion_type

        # 1. 初始卷积
        self.initial_conv = nn.Conv3d(in_channels, base_channels,
                                      kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm3d(base_channels)

        # 多尺度提取模块
        self.multiscale1 = MultiScaleFeatureExtractor(base_channels, base_channels * 2)
        self.downsample1 = nn.MaxPool3d(2, stride=2)

        self.multiscale2 = MultiScaleFeatureExtractor(base_channels * 2, base_channels * 4)
        self.downsample2 = nn.MaxPool3d(2, stride=2)

        self.multiscale3 = MultiScaleFeatureExtractor(base_channels * 4, base_channels * 8)

        # 2. 注意力机制
        self.cbam1 = CBAM3D(base_channels * 2)
        self.cbam2 = CBAM3D(base_channels * 4)
        self.cbam3 = CBAM3D(base_channels * 8)

        # 3. Transformer路径（全局特征）
        self.global_path = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        # 位置编码
        self.pos_encoding = PositionalEncoding3D(base_channels * 2)

        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock3D(base_channels * 2, num_heads=8) for _ in range(2)
        ])

        # 4. 注意力融合模块（替代原来的GlobalLocalFusion）
        fusion_dim = base_channels * 4

        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                global_dim=base_channels * 2,
                local_dim=base_channels * 8,
                fusion_dim=fusion_dim,
                num_heads=8
            )
        elif fusion_type == 'spatial_aware':
            self.fusion = SpatialAwareAttentionFusion(
                global_dim=base_channels * 2,
                local_dim=base_channels * 8,
                fusion_dim=fusion_dim,
                num_heads=8
            )
        elif fusion_type == 'multihead':
            self.fusion = MultiHeadAttentionFusion(
                global_dim=base_channels * 2,
                local_dim=base_channels * 8,
                fusion_dim=fusion_dim,
                num_heads=8
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # 5. 分类头
        self.final_proj = nn.Linear(fusion_dim, base_channels * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(base_channels * 2),
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_channels, num_classes)
        )

        # 辅助分类器
        self.aux_pool = nn.AdaptiveAvgPool3d(1)
        self.aux_classifier = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 4, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
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
        # 输入: (batch, channel, depth, height, width)

        # === 局部特征路径 ===
        x_local = F.relu(self.initial_bn(self.initial_conv(x)))

        x_local = self.multiscale1(x_local)
        x_local = self.cbam1(x_local)
        x_local = self.downsample1(x_local)

        x_local = self.multiscale2(x_local)
        x_local = self.cbam2(x_local)
        x_local = self.downsample2(x_local)

        x_local = self.multiscale3(x_local)
        x_local = self.cbam3(x_local)

        # === 全局特征路径 ===
        x_global = self.global_path(x_local)
        x_global = self.pos_encoding(x_global)

        b, c, d, h, w = x_global.shape
        x_global_seq = rearrange(x_global, 'b c d h w -> b (d h w) c')

        for transformer in self.transformer_blocks:
            x_global_seq = transformer(x_global_seq)

        # 全局池化
        global_pooled = x_global_seq.mean(dim=1)  # (B, base_channels*2)

        # === 注意力融合 ===
        fused = self.fusion(global_pooled, x_local)  # (B, fusion_dim)

        # === 分类 ===
        final_feat = self.final_proj(fused)
        main_output = self.classifier(final_feat)

        # 辅助输出（可选）
        aux_feat = self.aux_pool(x_local).squeeze(-1).squeeze(-1).squeeze(-1)
        aux_output = self.aux_classifier(aux_feat)

        return main_output, aux_output


# 创建模型的辅助函数
def create_attention_fusion_model(config=None):
    """创建使用注意力融合的模型实例"""
    if config is None:
        config = {
            'in_channels': 1,
            'base_channels': 32,
            'num_classes': 2,
            'input_shape': (64, 128, 128),
            'fusion_type': 'cross_attention'  # 可选: 'cross_attention', 'spatial_aware', 'multihead'
        }

    model = Medical3DClassifierWithAttentionFusion(
        in_channels=config['in_channels'],
        base_channels=config['base_channels'],
        num_classes=config['num_classes'],
        input_shape=config['input_shape'],
        fusion_type=config['fusion_type']
    )

    return model


# 使用示例
if __name__ == "__main__":
    # 创建不同融合策略的模型
    configs = [
        {'fusion_type': 'cross_attention', 'name': 'CrossAttention'},
        {'fusion_type': 'spatial_aware', 'name': 'SpatialAware'},
        {'fusion_type': 'multihead', 'name': 'MultiHead'}
    ]

    # 测试每个模型
    for cfg in configs:
        print(f"\n创建 {cfg['name']} 融合模型...")
        model_config = {
            'in_channels': 1,
            'base_channels': 32,
            'num_classes': 2,
            'input_shape': (64, 128, 128),
            'fusion_type': cfg['fusion_type']
        }

        model = create_attention_fusion_model(model_config)

        # 测试前向传播
        test_input = torch.randn(2, 1, 64, 128, 128)
        main_out, aux_out = model(test_input)

        print(f"主输出形状: {main_out.shape}")
        print(f"辅助输出形状: {aux_out.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")