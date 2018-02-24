
from torch import nn
from torch.nn import functional as F


class ConvolutionalLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.bn(x_in)
        x = self.conv(x)
        x = self.relu(x)
        return x


class SimpleConvolutionalAutoencoder(nn.Module):
    def __init__(self, image_width, image_height, image_channels,
            conv_channels, conv_bias=False):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height

        self.conv1 = ConvolutionalLayer(
                in_channels=image_channels,
                out_channels=conv_channels[0],
                bias=conv_bias)

        self.conv2 = ConvolutionalLayer(
                in_channels=conv_channels[0],
                out_channels=conv_channels[1],
                bias=conv_bias)

        self.conv3 = ConvolutionalLayer(
                in_channels=conv_channels[1],
                out_channels=conv_channels[2],
                padding=2,
                bias=conv_bias)

        self.bn4 = nn.BatchNorm2d(num_features=conv_channels[2])
        self.deconv1 = nn.ConvTranspose2d(in_channels=conv_channels[2],
                out_channels=conv_channels[1],
                stride=2,
                kernel_size=3,
                padding=2,
                output_padding=0,
                bias=conv_bias)

        self.bn5 = nn.BatchNorm2d(num_features=conv_channels[1])
        self.deconv2 = nn.ConvTranspose2d(in_channels=conv_channels[1],
                out_channels=conv_channels[0],
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=conv_bias)

        self.bn6 = nn.BatchNorm2d(num_features=conv_channels[0])
        self.deconv3 = nn.ConvTranspose2d(in_channels=conv_channels[0],
                out_channels=image_channels,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=conv_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x_in, **kwargs):
        x = self.conv1(x_in)
        x = self.conv2(x)
        x_latent = self.conv3(x)
        return x_latent

    def decode(self, x_latent):
        x = self.bn4(x_latent)
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.bn5(x)
        x = self.deconv2(x)
        x = self.relu(x)

        x = self.bn6(x)
        x = self.deconv3(x)
        x_out = self.sigmoid(x)
        return x_out

    def forward(self, x_in, **kwargs):
        x_latent = self.encode(x_in)
        x_out = self.decode(x_latent)
        return {'x_out': x_out}


def basic_autoencoder_loss_fn(x_out, x_in, **kwarg):
    return F.binary_cross_entropy(x_out, x_in.view_as(x_out),
            size_average=False)
