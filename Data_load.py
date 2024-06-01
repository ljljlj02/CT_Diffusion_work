import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def load_nifti_image(file_path):
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data.astype(np.float32)
    return torch.tensor(img_data)

class CTNiftiDataset(Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, transform=None):
        self.noisy_image_paths = noisy_image_paths
        self.clean_image_paths = clean_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, idx):
        noisy_img = load_nifti_image(self.noisy_image_paths[idx])
        clean_img = load_nifti_image(self.clean_image_paths[idx])
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        return noisy_img, clean_img

# 示例：加载数据集路径
noisy_image_paths = ['path/to/noisy_image1.nii', 'path/to/noisy_image2.nii', ...]
clean_image_paths = ['path/to/clean_image1.nii', 'path/to/clean_image2.nii', ...]
