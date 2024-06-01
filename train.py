import torch.optim as optim


def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_imgs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
dataset = CTNiftiDataset(noisy_image_paths, clean_image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

train_model(model, dataloader, criterion, optimizer, num_epochs=25)
