import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import random
from scipy import ndimage


class NiftiDataset(Dataset):
    def __init__(self, main_folder, target_shape=(64, 256, 256), transform=None, augment=False, mode='train'):

        self.main_folder = main_folder
        self.transform = transform
        self.shape = target_shape
        self.image_paths = []
        self.labels = []
        self.augment = augment
        self.mode = mode

        # 定义文件夹名称和对应的标签
        class_folders = {
            'abnormal': 1,
            'normal': 0
        }

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

        # 基本预处理
        image_data = self.resize_to_target(image_data, target_shape=self.shape)
        image_data = self.normalize_to_minus_one_one(image_data)

        # 添加通道维度 (C, D, H, W)
        image_data = np.expand_dims(image_data, axis=0)

        # 转换为tensor
        image_tensor = torch.from_numpy(image_data).float()

        # 获取标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 数据增强（仅训练时）
        if self.augment and self.mode == 'train':
            image_tensor = self.apply_3d_augmentations(image_tensor)

        # 应用额外的变换
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return {
            'image': image_tensor,
            'label': label,
            'path': self.image_paths[idx]
        }

    def apply_3d_augmentations(self, image):
        """
        应用3D数据增强，仿照文献中的Method III
        """
        # 随机旋转 (±15度)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            # 对3D图像应用旋转（绕z轴旋转）
            image = self.rotate_3d(image, angle)

        # 随机水平翻转
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])  # 沿宽度轴翻转

        # 随机对比度/亮度调整（模拟ColorJitter）
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            image = image * contrast + (brightness - 1)
            image = torch.clamp(image, -1, 1)

        # 随机裁剪并缩放（模拟RandomResizedCrop）
        if random.random() > 0.5:
            image = self.random_3d_crop(image, scale=(0.7, 1.0))

        return image

    @staticmethod
    def rotate_3d(image, angle):
        """
        对3D图像的每个切片应用2D旋转
        """
        if angle == 0:
            return image

        image_np = image.numpy()
        rotated = ndimage.rotate(image_np, angle, axes=(-2, -1),
                                 reshape=False, order=1, mode='nearest')
        return torch.from_numpy(rotated).float()

    def random_3d_crop(self, image, scale=(0.7, 1.0)):
        """
        随机裁剪并缩放到原尺寸
        """
        _, D, H, W = image.shape
        scale_factor = random.uniform(*scale)

        new_D = int(D * scale_factor)
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)

        # 随机起始点
        d_start = random.randint(0, D - new_D)
        h_start = random.randint(0, H - new_H)
        w_start = random.randint(0, W - new_W)

        # 裁剪
        cropped = image[:, d_start:d_start + new_D,
                  h_start:h_start + new_H,
                  w_start:w_start + new_W]

        # 重新缩放到原始尺寸
        resized = F.interpolate(cropped.unsqueeze(0),
                                size=(D, H, W),
                                mode='trilinear',
                                align_corners=False)
        return resized.squeeze(0)

    @staticmethod
    def resize_to_target(image, target_shape=(64, 256, 256)):
        current_shape = image.shape

        # 如果当前尺寸已经等于目标尺寸，直接返回
        if current_shape == target_shape:
            return image

        # 使用三线性插值调整尺寸
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            image_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        return resized.squeeze().numpy()

    @staticmethod
    def normalize_to_minus_one_one(image):
        # 获取最小值和最大值
        min_val = np.min(image)
        max_val = np.max(image)

        # 避免除以零的情况
        if max_val == min_val:
            normalized = np.zeros_like(image)
        else:
            normalized = (image - min_val) / (max_val - min_val)
            normalized = normalized * 2 - 1
        return normalized


def to_float_tensor(x):
    """将tensor转换为float类型，用于transform"""
    return x.float()


def get_data_transforms(mode='train'):
    """
    定义数据增强和预处理变换
    mode: 'train', 'val', 'test'
    """
    if mode == 'train':
        # 训练时使用重型增强（Method III）
        transform = transforms.Compose([
            transforms.Lambda(to_float_tensor),
        ])
    else:
        # 验证/测试时仅基本预处理
        transform = transforms.Compose([
            transforms.Lambda(to_float_tensor),
        ])
    return transform


class Mixup3D:
    """
    3D版本的Mixup增强
    在训练循环中使用，不是在DataLoader中
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch_images, batch_labels):
        if self.alpha <= 0:
            return batch_images, batch_labels

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_images.size(0)

        # 随机打乱批次
        index = torch.randperm(batch_size).to(batch_images.device)

        # 混合图像
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]

        # 混合标签（软标签）
        labels_onehot = F.one_hot(batch_labels, num_classes=2).float()
        labels_shuffled = F.one_hot(batch_labels[index], num_classes=2).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled

        return mixed_images, mixed_labels


class CutMix3D:
    """
    3D版本的CutMix增强
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch_images, batch_labels):
        if self.alpha <= 0:
            return batch_images, batch_labels

        batch_size = batch_images.size(0)
        _, D, H, W = batch_images.shape[1:]

        # 随机打乱批次
        index = torch.randperm(batch_size).to(batch_images.device)

        # 生成混合系数
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机选择裁剪区域
        cut_ratio = np.sqrt(1. - lam)
        cut_d = int(D * cut_ratio)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # 随机起始点
        cx = np.random.randint(D)
        cy = np.random.randint(H)
        cz = np.random.randint(W)

        bbx1 = np.clip(cx - cut_d // 2, 0, D)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbz1 = np.clip(cz - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_d // 2, 0, D)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbz2 = np.clip(cz + cut_w // 2, 0, W)

        # 实际裁剪区域大小
        cut_d = bbx2 - bbx1
        cut_h = bby2 - bby1
        cut_w = bbz2 - bbz1

        # 计算实际的lam
        lam = 1 - (cut_d * cut_h * cut_w) / (D * H * W)

        # 应用CutMix
        mixed_images = batch_images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = \
            batch_images[index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]

        # 混合标签
        labels_onehot = F.one_hot(batch_labels, num_classes=2).float()
        labels_shuffled = F.one_hot(batch_labels[index], num_classes=2).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled

        return mixed_images, mixed_labels


def create_dataloaders(main_folder, target_shape=(64, 256, 256),
                       batch_size=4, num_workers=0, train_val_split=0.8,
                       augment=True):
    """
    创建数据加载器，支持重型数据增强
    """
    # 创建完整数据集
    full_dataset = NiftiDataset(main_folder, target_shape=target_shape,
                                augment=False, mode='full')

    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size

    # 随机划分
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 创建训练和验证数据集
    from torch.utils.data import Subset

    # 训练集（应用增强）
    train_dataset = Subset(full_dataset, train_indices)
    train_dataset.dataset.mode = 'train'
    train_dataset.dataset.augment = augment
    train_dataset.dataset.transform = get_data_transforms('train')

    # 验证集（不应用增强）
    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.mode = 'val'
    val_dataset.dataset.augment = False
    val_dataset.dataset.transform = get_data_transforms('val')

    print(f"train_set: {len(train_dataset)}, val_set: {len(val_dataset)}")

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


def visualize_augmentations(dataset, num_samples=3):
    """
    可视化数据增强效果
    """
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        # 原始图像
        original = dataset[0]['image']
        original_slice = original[0, original.shape[1] // 2, :, :]
        axes[i, 0].imshow(original_slice.numpy(), cmap='gray')
        axes[i, 0].set_title(f'Sample {i + 1} Original')
        axes[i, 0].axis('off')

        # 应用不同的增强
        for j, augment_name in enumerate(['Rotated', 'Flipped', 'Contrast', 'Cropped']):
            img = dataset[0]['image'].clone()
            if augment_name == 'Rotated':
                img = dataset.rotate_3d(img, angle=15)
            elif augment_name == 'Flipped':
                img = torch.flip(img, dims=[-1])
            elif augment_name == 'Contrast':
                img = img * 1.2 + 0.1
                img = torch.clamp(img, -1, 1)
            elif augment_name == 'Cropped':
                img = dataset.random_3d_crop(img, scale=(0.7, 0.8))

            slice_img = img[0, img.shape[1] // 2, :, :]
            axes[i, j + 1].imshow(slice_img.numpy(), cmap='gray')
            axes[i, j + 1].set_title(f'{augment_name}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建数据加载器（启用增强）
    train_loader, val_loader = create_dataloaders(
        main_folder='your_data_path',
        target_shape=(64, 256, 256),
        batch_size=4,
        augment=True
    )

    # 创建增强实例（用于训练循环）
    mixup = Mixup3D(alpha=1.0)
    cutmix = CutMix3D(alpha=1.0)

    # 训练循环示例
    for epoch in range(10):
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].cuda()
            labels = batch['label'].cuda()

            # 随机选择一种混合增强
            if random.random() > 0.5:
                # 使用Mixup
                mixed_images, mixed_labels = mixup(images, labels)
            else:
                # 使用CutMix
                mixed_images, mixed_labels = cutmix(images, labels)

            # 这里继续您的训练流程...
            # model_output = model(mixed_images)
            # loss = criterion(model_output, mixed_labels)

            print(f"Batch {batch_idx}: Images shape {images.shape}, Mixed shape {mixed_images.shape}")
            break
        break