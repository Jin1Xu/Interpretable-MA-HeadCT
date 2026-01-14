import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


class PngDataset2D(Dataset):
    """
    Source data is 2D PNG images:
      - Each file is expected to be a PNG image.
      - We convert it to Grayscale (L) to ensure it is (H, W).
      - Resize to target_shape (H, W) using bilinear interpolation.
      - Normalize to [-1, 1] using instance min-max normalization.
      - Return tensor as (C, H, W) where C=1.
    """

    def __init__(self, main_folder, target_shape=(256, 256), transform=None):
        self.main_folder = main_folder
        self.transform = transform
        self.target_shape = target_shape  # (H, W)
        self.image_paths = []
        self.labels = []

        class_folders = {"important": 1, "unimportant": 0}

        for class_name, label in class_folders.items():
            class_path = os.path.join(main_folder, class_name)

            if not os.path.exists(class_path):
                print(f"Warning: folder not found: {class_path}")
                continue

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(".png"):
                    self.image_paths.append(os.path.join(class_path, file_name))
                    self.labels.append(label)

        print(f"total: {len(self.image_paths)}")
        print(f"label: 1={self.labels.count(1)}, 0={self.labels.count(0)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        img_pil = Image.open(path).convert('L')
        
        image_data = np.array(img_pil).astype(np.float32)
        
        # Resize to (H, W)
        image_data = self.resize_to_target_2d(image_data, target_shape=self.target_shape)

        # Normalize to [-1, 1]
        image_data = self.normalize_to_minus_one_one(image_data)

        # Add channel dim -> (C, H, W) -> (1, H, W)
        image_data = np.expand_dims(image_data, axis=0)

        image_tensor = torch.from_numpy(image_data).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return {
            "image": image_tensor,  # (1, H, W)
            "label": label,
            "path": path
        }

    @staticmethod
    def resize_to_target_2d(image: np.ndarray, target_shape=(256, 256)) -> np.ndarray:
        """
        Resize 2D numpy array (H, W) -> target_shape (H, W) using bilinear interpolation.
        """
        if image.shape == target_shape:
            return image.astype(np.float32)

        # image is (H, W), make it (1, 1, H, W) for F.interpolate
        x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
        return x.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def normalize_to_minus_one_one(image: np.ndarray) -> np.ndarray:
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if max_val == min_val:
            return np.zeros_like(image, dtype=np.float32)
        x = (image - min_val) / (max_val - min_val)  # [0,1]
        return (x * 2.0 - 1.0).astype(np.float32)    # [-1,1]


def to_float_tensor(x):
    return x.float()


def get_data_transforms():
    return transforms.Compose([
        transforms.Lambda(to_float_tensor),
    ])


def create_dataloaders(
    main_folder,
    target_shape=(256, 256),
    batch_size=16,
    num_workers=0,
    train_val_split=0.8,
    seed=42
):
    transform = get_data_transforms()
    full_dataset = PngDataset2D(main_folder, target_shape=target_shape, transform=transform)

    dataset_size = len(full_dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size

    indices = list(range(dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"train_set: {len(train_dataset)}, val_set: {len(val_dataset)}")

    if num_workers > 0 and os.name == "nt":
        print("Warning: Windows multi-worker may have issues. Setting num_workers=0.")
        num_workers = 0

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


if __name__ == "__main__":
    # 测试代码
    # 请确保路径下有 normal/abnormal 文件夹且包含 png 图片
    main_folder = "autodl-tmp/datasets"
    
    train_loader, val_loader = create_dataloaders(
        main_folder=main_folder,
        target_shape=(256, 256),
        batch_size=8,
        num_workers=0
    )

    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"Batch Shape: Image={batch['image'].shape}, Label={batch['label'].shape}")
        print(f"Value Range: Min={batch['image'].min():.2f}, Max={batch['image'].max():.2f}")