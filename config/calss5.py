import torch

num_epochs = 200
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-5

dataset = "/zjs/brain_ct/dataset"
test_dataset = "/zjs/brain_ct/dataset/test"
save_dir = "/zjs/brain_ct/brain_ct/result"
model_name = "multiclass_3d"
fusion_type = 'multihead'

in_channels = 1
base_channels = 32
num_classes = 5
input_shape = (96, 256, 256)

max_size_mb=3000
min_size_mb=0
train_val_split=0.8




