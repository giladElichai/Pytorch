import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=1, use_bn=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=(not use_bn))
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=1, output_padding=1, use_bn=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding, bias=(not use_bn))
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x   


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=False):
        super().__init__()

        self.downscale = downscale
        stride = 2 if downscale else 1 
        self.conv1 = ConvBlock(in_channels, out_channels, kernel=3, stride=stride, padding=1, use_bn=True)
        self.conv2 = ConvBlock(out_channels, out_channels,kernel=3, stride=1, padding=1, use_bn=True, activation=None)     
        if self.downscale:
            self.downscale_conv = ConvBlock(in_channels, out_channels, kernel=1, stride=stride, padding=0, use_bn=True, activation=None)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downscale:
            residual = self.downscale_conv(residual)
        x = x+residual
        x = torch.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock1 = ResidualBlock(in_channels, out_channels, downscale=True)
        self.resblock2 = ResidualBlock(out_channels, out_channels, downscale=False)      
    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x


class LinknetEncoder(nn.Module):
    def __init__(self, init_filters=64, n_levels=4):
        super().__init__()
        
        self.layers = nn.ModuleList()

        in_channels = init_filters
        for level in range(n_levels):
            out_channels = init_filters* 2**level
            self.layers.append(EncoderBlock(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            skip_connections.append(x)
        return x, skip_connections



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, in_channels//4, kernel=3, stride=1, padding=1)
        self.upscale = ConvTransposeBlock(in_channels//4, in_channels//4, kernel=3, stride=2, padding=1, output_padding=1)
        self.conv2 = ConvBlock(in_channels//4, out_channels, kernel=3, stride=1, padding=1)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.upscale(x)
        x = self.conv2(x)
        return x


class LinknetDecoder(nn.Module):
    def __init__(self, init_filters=64, n_levels=4):
        super().__init__()
        
        self.layers = nn.ModuleList()

        in_channels = init_filters * 2**(n_levels-1)
        for level in reversed(range(n_levels)):
            out_channels = init_filters* 2**max((level-1),0)
            self.layers.append(DecoderBlock(in_channels, int(out_channels)))
            in_channels = out_channels

    def forward(self, x, skip_connections):
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                skip_con = skip_connections[i+1]
                x = x + skip_con
        return x


class LinknetClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upscale = ConvTransposeBlock(in_channels, in_channels*2, kernel=3, stride=2, padding=1, output_padding=1)
        self.conv1 = ConvBlock(in_channels*2, in_channels*2, kernel=3, stride=1, padding=1)
        self.upscale_out = nn.ConvTranspose2d(in_channels*2, out_channels, 2, 2, 0)
        self.dropout = nn.Dropout2d(0.5)
        
    def forward(self, x):
        
        x = self.upscale(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.upscale_out(x)
        return x


class LinknetModel(nn.Module):

    def __init__(self, in_channels, num_classes, init_filters=64, n_levels=4 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_filters=init_filters
        self.n_levels = n_levels

        self.conv = ConvBlock(in_channels, init_filters, kernel=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, 2, padding=1)

        self.encoder = LinknetEncoder(init_filters=64, n_levels=4)
        self.decoder = LinknetDecoder(init_filters=64, n_levels=4)
        self.classifier = LinknetClassifier(init_filters, num_classes)
        self.dropout_b = nn.Dropout2d(0.5)
        self.dropout_u = nn.Dropout2d(0.5)


    def forward(self, x):
        
        x = self.conv(x)
        x = self.pool(x)
        x, skip_connections = self.encoder(x)
        x = self.dropout_b(x)
        skip_connections.reverse()
        x = self.decoder(x, skip_connections[:])
        x = self.dropout_u(x)
        x = self.classifier(x)
        return x 


def main():
    import numpy as np
    model = LinknetModel(3, 10)

    x = np.random.rand(4,3,256,256)
    x = torch.from_numpy(x).float()
    y = model(x)
    y = y.argmax(1).numpy()
    x = 0


if __name__ == "__main__":
    main()