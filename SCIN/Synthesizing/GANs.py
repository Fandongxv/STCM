import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


file_path = "./data/LendingClub_data/anomalies.csv"
df = pd.read_csv(file_path)
print(df.shape)
df=df.drop("mo_sin_old_il_acct",axis=1)
print("drop:",df.shape)
#print(df.info())
#print(data.info())
#print(pd.isnull(data).any())

X=df.values
#print(X.shape)

dataset = TensorDataset(torch.FloatTensor(X))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(62, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


generator = Generator()
discriminator = Discriminator()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()


epochs = 1
latent_dim = 100

for epoch in range(epochs):
    d_loss_sum = 0
    g_loss_sum = 0
    num_batches = 0

    for batch in dataloader:
        real_data = batch[0]

        
        fake_data = generator(torch.randn(real_data.size(0), latent_dim)).detach()
        real_output = discriminator(real_data)
        fake_output = discriminator(fake_data)

        d_loss_real = criterion(real_output, torch.ones_like(real_output))
        d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        
        z = torch.randn(real_data.size(0), latent_dim)
        fake_data = generator(z)
        fake_output = discriminator(fake_data)

        g_loss = criterion(fake_output, torch.ones_like(fake_output))

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        
        d_loss_sum += d_loss.item()
        g_loss_sum += g_loss.item()
        num_batches += 1

    
        print(f"Epoch: {epoch}, D_loss: {d_loss_sum / num_batches}, G_loss: {g_loss_sum / num_batches}")



num_samples_to_generate = 350260
generated_samples = []

for _ in range(num_samples_to_generate):
    z = torch.randn(latent_dim)
    generated_sample = generator(z)
    generated_samples.append(generated_sample.detach().numpy())

generated_samples = np.array(generated_samples)

print(generated_samples.shape)
reconstructed_df = pd.DataFrame(generated_samples)
reconstructed_df.columns=df.columns

reconstructed_df.to_csv("./data/LendingClub_data/GANs/reconstructed_GANs_lendingclub.csv", index=False)