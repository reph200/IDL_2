import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (batch, 1, 28, 28) -> (batch, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (batch, 16, 14, 14) -> (batch, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # (batch, 32, 7, 7) -> (batch, 64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class LatentSpace(nn.Module):
    def __init__(self, input_dim=64, latent_dim=12):
        super(LatentSpace, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 64, 1, 1)  # Reshape back to (batch, 64, 1, 1)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # (batch, 64, 1, 1) -> (batch, 32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # (batch, 32, 7, 7) -> (batch, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            # (batch, 16, 14, 14) -> (batch, 1, 28, 28)
            nn.Sigmoid()  # To make sure output is between 0 and 1
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.latent_space = LatentSpace()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent_space(x)
        x = self.decoder(x)
        return x