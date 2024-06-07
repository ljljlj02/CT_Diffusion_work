import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
from tqdm import tqdm

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps):
        super(DiffusionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t = t / self.num_timesteps
        t = t.view(-1, 1, 1, 1)
        t = t.repeat(1, x.size(1), x.size(2), x.size(3))
        x = torch.cat((x, t), dim=1)
        return self.net(x)

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

def add_noise(images, alpha):
    noisy_images = []
    for img in images:
        noise = np.random.normal(scale=np.sqrt(1 - alpha), size=img.shape)
        noisy_img = np.sqrt(alpha) * img + noise
        noisy_images.append(noisy_img)
    return np.array(noisy_images)

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def forward_diffusion(x0, timesteps, alpha):
    noise_levels = np.linspace(0, 1, timesteps)
    noisy_images = []
    for level in noise_levels:
        noise = np.random.normal(scale=np.sqrt(1 - level), size=x0.shape)
        noisy_img = np.sqrt(level) * x0 + noise
        noisy_images.append(noisy_img)
    return noisy_images

def backward_denoising(model, noisy_images, timesteps, alpha):
    x = noisy_images[-1]
    for t in reversed(range(timesteps)):
        x = model(x, torch.tensor([t], dtype=torch.float32).cuda())
    return x
