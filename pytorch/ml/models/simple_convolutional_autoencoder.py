
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


class DeconvolutionalLayer(nn.Module):
    def __init__(self, *, in_channels, out_channels,
            stride=2, kernel_size=3, padding=1,
            output_padding=1, bias=False, activation=nn.ReLU()):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                output_padding=output_padding,
                bias=bias)
        self.activation = activation

    def forward(self, x_in):
        x = self.bn(x_in)
        x = self.deconv(x)
        x = self.activation(x)
        return x


class SimpleConvolutionalAutoencoder(nn.Module):
    def __init__(self, image_channels, conv_channels, conv_bias=False):
        super().__init__()
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

        self.deconv1 = DeconvolutionalLayer(
                in_channels=conv_channels[2],
                out_channels=conv_channels[1],
                padding=2,
                output_padding=0,
                bias=conv_bias)

        self.deconv2 = DeconvolutionalLayer(
                in_channels=conv_channels[1],
                out_channels=conv_channels[0],
                bias=conv_bias)

        self.deconv3 = DeconvolutionalLayer(
                activation=nn.Sigmoid(),
                in_channels=conv_channels[0],
                out_channels=image_channels,
                bias=conv_bias)

    def encode(self, x_in, **kwargs):
        x = self.conv1(x_in)
        x = self.conv2(x)
        x_latent = self.conv3(x)
        return x_latent

    def decode(self, x_latent):
        x = self.deconv1(x_latent)
        x = self.deconv2(x)
        x_out = self.deconv3(x)
        return x_out

    def forward(self, x_in, **kwargs):
        x_latent = self.encode(x_in)
        x_out = self.decode(x_latent)
        return {'x_out': x_out}

    def calculate_losses(self, x_out, x_in, **kwargs):
        reconstruction_loss = F.binary_cross_entropy(x_out, x_in.view_as(x_out),
                size_average=False) / (x_in.shape[-1] * x_in.shape[-2])
        return {'total_loss': reconstruction_loss / x_in.shape[0]}
