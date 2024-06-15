import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.images = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):  # Verifica que sea un archivo
                    self.images.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def save_processed_dataset(dataset, file_path):
        torch.save({'images': dataset.images, 'labels': dataset.labels, 'classes': dataset.classes}, file_path)
