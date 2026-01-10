import torch

num_epochs = 1
batch_size = 2 
learning_rate = 1e-4
weight_decay = 1e-5
patience = 10

dataset = "/zhangyongquan/yhq/dataset/brain"
save_dir = "/zhangyongquan/yhq/brain_ct/result"
model_name = "basemodel"

in_channels = 1
base_channels = 32
num_classes = 2
input_shape = (96, 256, 256)
