import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


class NiftiDataset(Dataset):
    def __init__(self, main_folder, target_shape = (96, 256, 256), transform=None):

        self.main_folder = main_folder
        self.transform = transform
        self.shape = target_shape
        self.image_paths = []
        self.labels = []

        class_folders = {
            'abnormal': 1,
            'normal': 0}

        # 遍历所有类别的文件夹
        for class_name, label in class_folders.items():
            class_path = os.path.join(main_folder, class_name)

            if not os.path.exists(class_path):
                print(f"警告: 文件夹 {class_path} 不存在")
                continue

            # 获取所有nii文件
            for file_name in os.listdir(class_path):
                if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                    file_path = os.path.join(class_path, file_name)
                    self.image_paths.append(file_path)
                    self.labels.append(label)

        print(f"total: {len(self.image_paths)} ")
        print(f"label: 1={self.labels.count(1)}, 0={self.labels.count(0)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取nii文件
        nii_img = nib.load(self.image_paths[idx])
        image_data = nii_img.get_fdata()

        # 转换为float32
        image_data = image_data.astype(np.float32)

        image_data = self.resize_to_target(image_data, target_shape=self.shape)

        # 预处理：归一化到[-1, 1]
        image_data = self.normalize_to_minus_one_one(image_data)

        # 添加通道维度 (C, D, H, W)
        image_data = np.expand_dims(image_data, axis=0)

        # 转换为tensor并转换为float
        image_tensor = torch.from_numpy(image_data).float()

        # 获取标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 应用额外的变换
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return {
            'image': image_tensor,
            'label': label,
            'path': self.image_paths[idx]
        }

    @staticmethod
    def resize_to_target(image, target_shape=(96, 256, 256)):
        current_shape = image.shape

        # 如果当前尺寸已经等于目标尺寸，直接返回
        if current_shape == target_shape:
            return image

        # 使用三线性插值调整尺寸
        # 先将numpy数组转换为tensor以便使用torch的插值函数
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, D, H, W)

        # 使用三线性插值
        resized = F.interpolate(
            image_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )

        # 移除添加的维度并转换回numpy
        resized_image = resized.squeeze().numpy()

        return resized_image

    @staticmethod
    def normalize_to_minus_one_one(image):
        # 获取最小值和最大值
        min_val = np.min(image)
        max_val = np.max(image)

        # 避免除以零的情况
        if max_val == min_val:
            # 如果所有值都相同，设置为0
            normalized = np.zeros_like(image)
        else:
            # 归一化到[0, 1]
            normalized = (image - min_val) / (max_val - min_val)
            # 映射到[-1, 1]
            normalized = normalized * 2 - 1
        return normalized


def to_float_tensor(x):
    """将tensor转换为float类型，用于transform"""
    return x.float()


def get_data_transforms():
    """定义数据增强和预处理变换"""
    transform = transforms.Compose([
        transforms.Lambda(to_float_tensor),
    ])
    return transform


def create_dataloaders(main_folder, target_shape=(96, 256, 256), batch_size=4, num_workers=0, train_val_split=0.8):
    # 创建数据集
    transform = get_data_transforms()
    full_dataset = NiftiDataset(main_folder, target_shape=target_shape, transform=transform)

    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size

    # 随机划分
    indices = list(range(dataset_size))
    np.random.seed(42)  # 设置随机种子确保可重复性
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 创建子数据集
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"train_set: {len(train_dataset)}, val_set: {len(val_dataset)}")

    # 注意：Windows上使用多进程可能导致问题，建议将num_workers设为0
    if num_workers > 0 and os.name == 'nt':  # 'nt' 表示Windows
        print("警告：在Windows上使用多进程可能导致序列化问题，将num_workers设为0")
        num_workers = 0

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader