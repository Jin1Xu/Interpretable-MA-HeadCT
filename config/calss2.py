import torch

num_epochs = 200
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-5
patience = 10

dataset = "/zjs/brain_ct/dataset"
save_dir = "/zjs/brain_ct/brain_ct/result"
model_name = "binary_loss"
fusion_type = 'multihead'

in_channels = 1
base_channels = 32
num_classes = 2
input_shape = (96, 256, 256)
max_size_mb=3000
min_size_mb=0
# 损失函数参数（新增）
loss_type = "weighted_ce"  # 损失函数类型
critical_weight = 3.0  # 危急值样本权重



