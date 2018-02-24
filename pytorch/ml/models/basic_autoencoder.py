from torch import nn
from torch.nn import functional as F


class BasicAutoencoder(nn.Module):
    def __init__(self, image_width, image_height, hidden_size, latent_size):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height

        self.bn1 = nn.BatchNorm1d(image_width * image_height)
        self.fc1 = nn.Linear(image_width * image_height, hidden_size)

        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)

        self.bn3 = nn.BatchNorm1d(latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)

        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, image_width * image_height)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x_in, **kwargs):
        encoder_input = x_in.view(-1, self.image_width * self.image_height)
        hidden_layer = self.relu(self.fc1(self.bn1(encoder_input)))
        x_latent = self.relu(self.fc2(self.bn2(hidden_layer)))
        return x_latent

    def decode(self, x_latent):
        hidden_layer = self.relu(self.fc3(self.bn3(x_latent)))
        x_out = self.sigmoid(self.fc4(self.bn4(hidden_layer)))
        return x_out

    def forward(self, x_in, **kwargs):
        x_latent = self.encode(x_in)
        x_out = self.decode(x_latent)
        return {'x_out': x_out}


def basic_autoencoder_loss_fn(x_out, x_in, **kwargs):
    return F.binary_cross_entropy(x_out, x_in.view_as(x_out),
            size_average=False)
