import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset

class CTDataset(Dataset):
    def __init__(self, noisy_images, clean_images):
        self.noisy_images = noisy_images
        self.clean_images = clean_images

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy = np.expand_dims(self.noisy_images[idx], axis=0)
        clean = np.expand_dims(self.clean_images[idx], axis=0)
        return torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32)

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def add_noise(images, alpha):
    noisy_images = []
    for img in images:
        noise = np.random.normal(scale=np.sqrt(1 - alpha), size=img.shape)
        noisy_img = np.sqrt(alpha) * img + noise
        noisy_images.append(noisy_img)
    return np.array(noisy_images)

clean_images = load_nifti('clean_images.nii.gz')
alpha = 0.5
timesteps = 100
noisy_images = add_noise(clean_images, alpha)

dataset = CTDataset(noisy_images, clean_images)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

