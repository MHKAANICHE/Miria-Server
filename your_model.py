import torch
import librosa
import numpy as np
import torch.nn as nn

class NoisingAutoencoder(nn.Module):
    def __init__(self):
        super(NoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        noised = self.decoder(encoded)
        return noised