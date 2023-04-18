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


# minority_data = df.drop('loan_status', axis=1)


X = df.values

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)


dataset = TensorDataset(torch.FloatTensor(X_normalized))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(62, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 62)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = torch.sigmoid(self.fc3(h))
        return out


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


latent_dim = 20
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)


optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)


epochs = 1

for epoch in range(epochs):
    for batch in dataloader:
        x = batch[0]

       
        mean, logvar = encoder(x)

        
        z = reparameterize(mean, logvar)

        
        x_reconstructed = decoder(z)

        
        reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_divergence_loss

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")


num_samples_to_generate = 350260
generated_samples = []

for _ in range(num_samples_to_generate):
    z = torch.randn(latent_dim)
    generated_sample = decoder(z)
    generated_samples.append(generated_sample.detach().numpy())


generated_samples = np.array(generated_samples)

generated_samples_original = scaler.inverse_transform(generated_samples)

print(generated_samples_original.shape)
reconstructed_df = pd.DataFrame(generated_samples_original)
reconstructed_df.columns=df.columns

reconstructed_df.to_csv("./data/LendingClub_data/VAEs/reconstructed_VAEs_lendingclub.csv", index=False)

