import torch

num_epochs = 200
batch_size = 8 
learning_rate = 1e-4
weight_decay = 1e-5
patience = 10

dataset = "new_datasets"
save_dir = "autodl-tmp/result"
model_name = "basemodel"

in_channels = 1
base_channels = 32
num_classes = 2
input_shape = (512, 512)
