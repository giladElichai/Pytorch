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



class Resblock(nn.Module):
    
    def __init__(self, in_channels, filters, downsample=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = filters
        
        stride = 2 if downsample else 1
        self.conv1 = ConvBlock(in_channels, filters, kernel=1, stride=stride)
        self.conv2 = ConvBlock(filters, filters, kernel=3, stride=1)
        self.conv3 = ConvBlock(filters, filters*4, kernel=1, stride=1)
        self.res_conv = None
        if downsample or filters != in_channels:
            self.res_conv = ConvBlock(filters, filters*4, kernel=1, stride=stride)
            
        self.relu = nn.ReLU()

                        
    def forward(self, x):
        
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.res_conv:
            residual = self.res_conv(residual)  
        return self.relu(x + residual)


class ResnetModel(nn.Module):

    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes\
            
        self.input_conv = ConvBlock(in_channels, 64, (7,7), stride=2, padding=3)
        self.pool = nn.MaxPool2d((3,3), stride=2, padding=1)

    def make_layer(in_channels, filters, downsample):
        pass
        


    def forward(self, x):
        
        x = self.input_conv(x)
        x = self.pool(x)

        return x 


def main():
    import numpy as np
    model = ResnetModel(3, 10)

    x = np.random.rand(4,3,224,224)
    x = torch.from_numpy(x).float()
    y = model(x)
    y = y.argmax(1).numpy()
    x = 0


if __name__ == "__main__":
    main()