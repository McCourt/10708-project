import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim as optim


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.main = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))

    def forward(self, x):
        x = self.pre(x)
        return x + self.main(x)


class GeneratorModel(nn.Module):
    def __init__(self, hidden_size=512, feature_size=40, z_dim=512):
        super(GeneratorModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.z_dim = z_dim
        total_size = self.z_dim + self.feature_size
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels=3, out_channels=64),
            nn.MaxPool2d(kernel_size=2), # batch x 16 x 32 x 32
            ResidualBlock(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2), # batch x 32 x 16 x 16
            ResidualBlock(in_channels=128, out_channels=256),
            nn.MaxPool2d(kernel_size=2), # batch x 64 x 8 x 8
            ResidualBlock(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2), # batch x 64 x 8 x 8
            nn.Conv2d(in_channels=256, out_channels=self.hidden_size, kernel_size=4, padding=0, bias=True),

            nn.Flatten(), # batch x 60,
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=self.hidden_size, out_features=self.z_dim * 2, bias=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=total_size, out_features=total_size, bias=True), # batch x 100 x 1 x 1
            nn.Unflatten(dim=1, unflattened_size=(total_size, 1, 1)),
            nn.ConvTranspose2d(in_channels=total_size, out_channels=128, stride=2, kernel_size=2, bias=True), # batch x 100 x 8 x 8
            nn.LeakyReLU(inplace=True),
            ResidualBlock(in_channels=128, out_channels=256), # batch x 64 x 2 x 2
            nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=2, bias=True), # batch x 100 x 8 x 8
            nn.LeakyReLU(inplace=True),
            ResidualBlock(in_channels=256, out_channels=256), # batch x 64 x 4 x 4
            nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=2, bias=True), # batch x 100 x 8 x 8
            nn.LeakyReLU(inplace=True),
            ResidualBlock(in_channels=256, out_channels=256), # batch x 64 x 8 x 8
            nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=2, bias=True),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(in_channels=256, out_channels=128), # batch x 32 x 16 x 16
            nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=2, bias=True),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(in_channels=128, out_channels=64), # batch x 16 x 32 x 32

            nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=2, kernel_size=2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        encoder_output = self.encoder(x)
        mu = encoder_output[..., :self.z_dim]
        log_std = encoder_output[..., self.z_dim:]
        std = torch.exp(log_std)
        
        if self.training:
            sampled_z = mu + torch.randn_like(std) * std
            x_hat = self.decoder(torch.cat([sampled_z, labels], dim=-1))
        else:
            x_hat = self.decoder(torch.cat([mu, labels], dim=-1))

        return (x_hat + 1) / 2, mu, std
    

class DiscriminatorModel(nn.Module):
    def __init__(self, hidden_size=512, feature_size=40):
        super(DiscriminatorModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.D = nn.Sequential(
            ResidualBlock(in_channels=3, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),
            nn.MaxPool2d(kernel_size=2), # batch x 16 x 32 x 32

            ResidualBlock(in_channels=16, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),
            nn.MaxPool2d(kernel_size=2), # batch x 32 x 16 x 16

            ResidualBlock(in_channels=32, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2), # batch x 64 x 8 x 8

            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2), # batch x 64 x 4 x 4

            nn.Conv2d(in_channels=64, out_channels=self.hidden_size, kernel_size=4, bias=True),
            nn.Flatten(), # batch x 60,
        )

        self.l1 = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=2 * self.hidden_size, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2 * self.hidden_size, out_features=1, bias=True),
            nn.Sigmoid()
        )

        self.l2 = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=2 * self.hidden_size, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2 * self.hidden_size, out_features=self.feature_size, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.D(x)
        return self.l1(f), self.l2(f)
