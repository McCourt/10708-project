import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim as optim


class GeneratorModel(nn.Module):
    def __init__(self, device, hidden_size=60, feature_size=40):
        super(GeneratorModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        total_size = self.hidden_size + self.feature_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2), # batch x 16 x 32 x 32

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2), # batch x 32 x 16 x 16

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2), # batch x 64 x 8 x 8
            nn.Conv2d(in_channels=64, out_channels=self.hidden_size, kernel_size=8, padding=0, bias=True),
            nn.Flatten(), # batch x 60,
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=total_size, out_features=total_size, bias=True), # batch x 100 x 1 x 1
            nn.Unflatten(dim=1, unflattened_size=(total_size, 1, 1)),
            nn.ConvTranspose2d(in_channels=total_size, out_channels=total_size, stride=2, kernel_size=2, bias=True), # batch x 100 x 8 x 8
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=total_size, out_channels=total_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=total_size), # batch x 64 x 2 x 2

            nn.ConvTranspose2d(in_channels=total_size, out_channels=total_size, stride=2, kernel_size=2, bias=True), # batch x 100 x 8 x 8
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=total_size, out_channels=total_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=total_size), # batch x 64 x 4 x 4

            nn.ConvTranspose2d(in_channels=total_size, out_channels=total_size, stride=2, kernel_size=2, bias=True), # batch x 100 x 8 x 8
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=total_size, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=64), # batch x 64 x 8 x 8

            nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=2, kernel_size=2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=32), # batch x 32 x 16 x 16

            nn.ConvTranspose2d(in_channels=32, out_channels=32, stride=2, kernel_size=2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=16), # batch x 16 x 32 x 32

            nn.ConvTranspose2d(in_channels=16, out_channels=16, stride=2, kernel_size=2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        encoder_output = self.encoder(x)
        mu = encoder_output[..., :self.hidden_size]
        logvar = encoder_output[..., self.hidden_size:]
        eps = torch.randn(mu.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        sampled_z = mu + eps * sigma
        x_hat = self.decoder(torch.cat([sampled_z, labels], dim=-1))

        return x_hat, mu, logvar
    

class ClassifierModel(nn.Module):
    def __init__(self, hidden_size=512, feature_size=40):
        super(ClassifierModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.D = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2), # batch x 16 x 32 x 32

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2), # batch x 32 x 16 x 16

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2), # batch x 64 x 8 x 8

            nn.Conv2d(in_channels=64, out_channels=self.hidden_size, kernel_size=8, padding=0, bias=True),
            nn.Flatten(), # batch x 60,
        )

        self.l2 = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=2 * self.hidden_size, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2 * self.hidden_size, out_features=self.feature_size, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        f = self.D(x)
        return self.l2(f)


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
