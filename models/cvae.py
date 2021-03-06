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
    def __init__(self, hidden_size=128, feature_size=40):
        super(GeneratorModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.total_size = self.hidden_size + self.feature_size
        self.ones = torch.ones([1, 1, 64, 64]).to('cuda')
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3 + self.feature_size, out_channels=32, kernel_size=4, stride=2, padding=1), # 32 * 32 * 32
            nn.BatchNorm2d(32),
            nn.GELU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), # 64 * 16 * 16
            nn.BatchNorm2d(64),
            nn.GELU(), 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), # 128 * 8 * 8
            nn.BatchNorm2d(128),
            nn.GELU(), 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), # 256 * 4 * 4
            nn.BatchNorm2d(256),
            nn.GELU(), 
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), # 512 * 2 * 2
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1), # 1024 * 1 * 1
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=self.hidden_size * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.total_size, out_features=1024), 
            nn.Unflatten(dim=1, unflattened_size=(1024, 1, 1)), # batch x 1024 x 1 x 1
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, stride=2, kernel_size=4, padding=1), # batch x 512 x 2 x 2
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1), # batch x 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1), # batch x 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=4, padding=1), # batch x 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=4, padding=1), # batch x 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, stride=2, kernel_size=4, padding=1), # batch x 3 x 64 x 64
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        labels_map = self.ones * labels.unsqueeze(-1).unsqueeze(-1)
        encoder_input = torch.cat([x, labels_map], dim=1)
        encoder_output = self.encoder(encoder_input)
        mu = encoder_output[..., :self.hidden_size]
        logvar = encoder_output[..., self.hidden_size:]
        sigma = torch.exp(logvar / 2)

        if self.training:
            eps = torch.randn_like(sigma)
            sampled_z = mu + eps * sigma
            x_hat = self.decoder(torch.cat([sampled_z, labels], dim=-1))
        else:
            x_hat = self.decoder(torch.cat([mu, labels], dim=-1))

        return x_hat, mu, logvar

class PriorModel(nn.Module):
    def __init__(self, hidden_size=60, feature_size=40):
        super(PriorModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.total_size = self.hidden_size + self.feature_size

        self.encoder = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=self.hidden_size * 2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size * 2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size * 2, bias=True)
        )
    
    def forward(self, labels):
        encoder_output = self.encoder(labels)
        mu = encoder_output[..., :self.hidden_size]
        logvar = encoder_output[..., self.hidden_size:]
        sigma = torch.exp(logvar / 2)

        if self.training:
            eps = torch.randn_like(sigma)
            sampled_z = mu + eps * sigma
        else:
            sampled_z = mu

        return sampled_z, mu, logvar

class ClassifierModel(nn.Module):
    def __init__(self, hidden_size=128, feature_size=40):
        super(ClassifierModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.classifier = nn.Sequential(
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

            nn.Conv2d(in_channels=64, out_channels=2 * self.feature_size, kernel_size=4, bias=True),
            nn.Flatten(), 
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2 * self.feature_size, out_features=self.feature_size, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('device:', device)
#
# # data
# training_parameters = {
#     "n_epochs": 100,
#     "batch_size": 100,
# }
# data_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=True, download=True,
#                     transform=transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.5,), (0.5,))
#                     ])),
#     batch_size=training_parameters["batch_size"],
#     shuffle=True
# )
#
# discriminator = DiscriminatorModel()
# generator = GeneratorModel()
# discriminator.to(device)
# generator.to(device)
#
# loss = nn.BCELoss()
# discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
# generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
#
# batch_size = training_parameters["batch_size"]
# n_epochs = training_parameters["n_epochs"]
# # train loop
# for epoch_idx in range(n_epochs):
#     G_loss = []
#     D_loss = []
#     for batch_idx, data_input in enumerate(data_loader):
#         '''
#         training for discriminator
#         '''
#         # Generate noise and move it the device
#         noise = torch.randn(batch_size, 100).to(device)
#         fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
#         # Forward pass for the fake and real images
#         generated_data = generator(noise, fake_labels) # batch_size X 784
#         true_data = data_input[0].view(batch_size, 784).to(device) # batch_size X 784
#         digit_labels = data_input[1] # batch_size
#         true_labels = torch.ones(batch_size).to(device) # all 1s
#
#         # Clear optimizer gradients
#         discriminator_optimizer.zero_grad()
#         # Forward pass with true data as input
#         discriminator_output_for_true_data = discriminator(true_data, digit_labels).view(batch_size)
#         # Compute Loss
#         true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)
#         # Forward pass with generated data as input
#         discriminator_output_for_generated_data = discriminator(generated_data.detach(), fake_labels).view(batch_size)
#         # Compute Loss
#         generator_discriminator_loss = loss(
#             discriminator_output_for_generated_data, torch.zeros(batch_size).to(device)
#         )
#         # Average the loss
#         discriminator_loss = (
#             true_discriminator_loss + generator_discriminator_loss
#         ) / 2
#
#         # Backpropagate the losses for Discriminator model
#         discriminator_loss.backward()
#         discriminator_optimizer.step()
#         D_loss.append(discriminator_loss.data.item())
#
#         '''
#         training for generator
#         '''
#         # Clear optimizer gradients
#         generator_optimizer.zero_grad()
#         # generate the data again
#         generated_data = generator(noise, fake_labels) # batch_size X 784
#         # Forward pass with the generated data
#         discriminator_output_on_generated_data = discriminator(generated_data, fake_labels).view(batch_size)
#         # Compute loss
#         generator_loss = loss(discriminator_output_on_generated_data, true_labels)
#         # Backpropagate losses for Generator model.
#         generator_loss.backward()
#         generator_optimizer.step()
#         G_loss.append(generator_loss.data.item())
#
#         # Evaluate the model
# #         if ((batch_idx + 1)% 500 == 0 and (epoch_idx + 1)%10 == 0):
# #             print("Training Steps Completed: ", batch_idx)
# #             with torch.no_grad():
# #                 noise = torch.randn(batch_size,100).to(device)
# #                 fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
# #                 generated_data = generator(noise, fake_labels).cpu().view(batch_size, 28, 28)
# #                 for x in generated_data:
# #                     print(fake_labels[0].item())
# #                     plt.imshow(x.detach().numpy(), interpolation='nearest',cmap='gray')
# #                     plt.show()
# #                     break
#
#     print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
#             (epoch_idx), n_epochs, torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss))))
