import torch.optim as optim
from tqdm import tqdm

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0
    for noisy, clean in tqdm(dataloader):
        noisy, clean = noisy.cuda(), clean.cuda()
        optimizer.zero_grad()

        batch_size = noisy.size(0)
        t = torch.randint(0, timesteps, (batch_size,), dtype=torch.float32).cuda()
        output = model(noisy, t)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

