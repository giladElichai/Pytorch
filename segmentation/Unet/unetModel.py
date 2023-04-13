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


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=1, us_bn=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel, stride, padding, us_bn, activation)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel, stride, padding, us_bn, activation)      
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, prms=[64,128,256,512]):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for prm in prms:
            self.layers.append(DoubleConvBlock(in_channels, prm, kernel=3))
            in_channels = prm

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            skip_connections.append(x)
            x = torch.max_pool2d(x, kernel_size=2, stride=2)
        return x, skip_connections


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_mode="inerpolate"):
        super().__init__()
        
        if upscale_mode == "inerpolate":
            self.upscale = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel=3),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
        else:
            self.upscale = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        self.conv = DoubleConvBlock(in_channels, out_channels, kernel=3)
        
    def forward(self, x, skip_con):
        x = self.upscale(x)
        x = torch.relu(x)
        x = torch.concat([skip_con, x], dim=1)
        x = self.conv(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, in_channels, prms=reversed([64,128,256,512]), upscale_mode="convtranspose"):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for prm in prms:
            self.layers.append(DecoderLayer(in_channels, prm, upscale_mode=upscale_mode))
            in_channels = prm

    def forward(self, x, skip_connections):
        
        for i, layer in enumerate(self.layers):
            skip_con = skip_connections[i]
            x = layer(x, skip_con)
        return x


class UnetModel(nn.Module):

    def __init__(self, in_channels, num_classes, prms=[64,128,256,512], upscale_mode="convtranspose") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.prms=prms

        self.encoder = UnetEncoder(in_channels, prms)
        self.bottom = DoubleConvBlock(prms[-1], prms[-1]*2, kernel=3)
        self.decoder = UnetDecoder(prms[-1]*2, reversed(prms), upscale_mode)
        self.classifier = nn.Conv2d(prms[0], num_classes, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(0.5)
        # ConvBlock(prms[0], num_classes, kernel=1, padding=0, activation=None)


    def forward(self, x):
        
        x, skip_connections = self.encoder(x)
        skip_connections.reverse()
        x = self.bottom(x)
        x = self.decoder(x, skip_connections)
        x = self.dropout(x)
        x = self.classifier(x)
        return x 


def main():
    import numpy as np
    model = UnetModel(3, 10)

    x = np.random.rand(4,3,256,256)
    x = torch.from_numpy(x).float()
    y = model(x)
    y = y.argmax(1).numpy()
    x = 0


if __name__ == "__main__":
    main()