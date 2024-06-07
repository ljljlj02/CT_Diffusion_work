import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps):
        super(DiffusionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t = t / self.num_timesteps
        t = t.view(-1, 1, 1, 1)
        t = t.repeat(1, x.size(2), x.size(3), 1).transpose(1, 3)
        x = torch.cat((x, t), dim=1)
        return self.net(x)

def forward_diffusion(x0, timesteps, alpha):
    noise_levels = np.linspace(0, 1, timesteps)
    noisy_images = []
    for level in noise_levels:
        noise = np.random.normal(scale=np.sqrt(1 - level), size=x0.shape)
        noisy_img = np.sqrt(level) * x0 + noise
        noisy_images.append(noisy_img)
    return noisy_images

def backward_denoising(model, noisy_images, timesteps):
    x = torch.tensor(noisy_images[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t], dtype=torch.float32).cuda()
        x = model(x, t_tensor)
    return x.squeeze().cpu().numpy()

model = DiffusionModel(num_timesteps=timesteps).cuda()

