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

class Resblock18(nn.Module):    
    def __init__(self, in_channels, filters, downsample=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = filters
        
        stride = 2 if downsample else 1
        self.conv1 = ConvBlock(in_channels, filters, kernel=3, stride=stride, padding=1)
        self.conv2 = ConvBlock(filters, filters, kernel=3, stride=1, padding=1, activation=None)
        self.res_conv = None
        if downsample or filters*4 != in_channels:
            self.res_conv = ConvBlock(in_channels, filters, kernel=1, stride=stride, padding=0, activation=None)
        self.relu = nn.ReLU()
             
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.res_conv:
            residual = self.res_conv(residual)  
        return self.relu(x + residual)



class Resblock(nn.Module):
    def __init__(self, in_channels, filters, downsample=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = filters*4
        
        stride = 2 if downsample else 1
        self.conv1 = ConvBlock(in_channels, filters, kernel=1, stride=stride, padding=0)
        self.conv2 = ConvBlock(filters, filters, kernel=3, stride=1)
        self.conv3 = ConvBlock(filters, filters*4, kernel=1, stride=1, padding=0, activation=None)
        self.res_conv = None
        if downsample or filters*4 != in_channels:
            self.res_conv = ConvBlock(in_channels, filters*4, kernel=1, stride=stride, padding=0, activation=None)
            
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
    def __init__(self, in_channels, num_classes=10, layer_type=50, layers_prms=[[3,64,True],]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer_type = layer_type
        self.block = Resblock
        self.expansion = 4
        if layer_type == 18:
            self.block = Resblock18
            self.expansion = 1
            
        self.input_conv = ConvBlock(in_channels, 64, (7,7), stride=2, padding=3)
        self.pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        in_channels = 64

        self.features_layers = nn.ModuleList()
        for layer_prm in layers_prms:   
            layer, in_channels = self.make_layer(in_channels=in_channels, num_blocks=layer_prm[0], filters=layer_prm[1], downsample=layer_prm[2])
            self.features_layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(layer_prm[1]*self.expansion, num_classes)

    def make_layer(self, in_channels, num_blocks, filters, downsample):
        layers = []
        for i in range(num_blocks):
            layers.append(self.block(in_channels, filters, downsample))
            downsample =  False
            in_channels = layers[-1].out_channels
        return nn.Sequential(*layers), in_channels
        
    def forward(self, x):
        
        x = self.input_conv(x)
        x = self.pool(x)
        for idx, layer in enumerate(self.features_layers):
            x = layer(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x 

def Resnet18(in_channels, num_classes):
    layers_prms = [[2,64,False],[2,128,True],[2,256,True],[2,512,True],]
    return ResnetModel(in_channels, num_classes, layer_type=18, layers_prms=layers_prms)

def Resnet34(in_channels, num_classes):
    layers_prms = [[3,64,False],[4,128,True],[6,256,True],[3,512,True],]
    return ResnetModel(in_channels, num_classes, layer_type=18, layers_prms=layers_prms)

def Resnet50(in_channels, num_classes):
    layers_prms = [[3,64,False],[4,128,True],[6,256,True],[3,512,True],]
    return ResnetModel(in_channels, num_classes, layer_type=50, layers_prms=layers_prms)

def Resnet101(in_channels, num_classes):
    layers_prms = [[3,64,False],[4,128,True],[23,256,True],[3,512,True],]
    return ResnetModel(in_channels, num_classes, layer_type=50, layers_prms=layers_prms)

def Resnet152(in_channels, num_classes):
    layers_prms = [[3,64,False],[4,128,True],[36,256,True],[3,512,True],]
    return ResnetModel(in_channels, num_classes, layer_type=50, layers_prms=layers_prms)

def main():
    import numpy as np
    model = Resnet18(3, 10)

    x = np.random.rand(4,3,224,224)
    x = torch.from_numpy(x).float()
    y = model(x)
    y = y.argmax(1).numpy()
    x = 0


if __name__ == "__main__":
    main()