from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
import torch
from model.DiWT import DiWTBlock
from functools import partial
from typing import Tuple
from model.resnet18 import ResNet

        

class MeanShift1(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4236, 0.4176, 0.3890), rgb_std=(1,1,1), sign=-1):

        super(MeanShift1, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class MeanShift2(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4417, 0.4346, 0.4008), rgb_std=(1,1,1), sign=-1):

        super(MeanShift2, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
            )

    def forward(self, x):
        x = self.up(x)
        return x
    
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
class DiWTBU(nn.Module):
    def __init__(self, 
                 input_size: Tuple[int, int],
                 basic_ch=64):
        super(DiWTBU, self).__init__()
        self.sizes = [(w//(2**i),h//(2**i)) for i in range(4) for w,h in [input_size]]
        self.DiWTBlock1 = DiWTBlock(
        in_channels = basic_ch,
        out_channels = basic_ch,
        squeeze_ratio = 0.25,
        expansion_ratio = 4,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99),
        activation_layer=nn.GELU,
        head_dim = 32,
        mlp_ratio = 4,
        mlp_dropout = 0,
        attention_dropout = 0,
        # partitioning parameters
        partition_size = 16,
        input_dilation_size = self.sizes[0],
        # number of layers
        n_layers = 2,
        p_stochastic = [0, 0],
        #p_stochastic = [0.2/10, 0]
)
        self.DiWTBlock2 = DiWTBlock(
        in_channels = basic_ch*2,
        out_channels = basic_ch*2,
        squeeze_ratio = 0.25,
        expansion_ratio = 4,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99),
        activation_layer=nn.GELU,
        head_dim = 32,
        mlp_ratio = 4,
        mlp_dropout = 0,
        attention_dropout = 0,
        # partitioning parameters
        partition_size = 16,
        input_dilation_size = self.sizes[1],
        # number of layers
        n_layers = 2,
        p_stochastic = [0, 0],
        #p_stochastic = [0.6/10, 0.4/10],
)
        self.DiWTBlock3 = DiWTBlock(
        in_channels = basic_ch*4,
        out_channels = basic_ch*4,
        squeeze_ratio = 0.25,
        expansion_ratio = 4,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99),
        activation_layer=nn.GELU,
        head_dim = 32,
        mlp_ratio = 4,
        mlp_dropout = 0,
        attention_dropout = 0,
        # partitioning parameters
        partition_size = 16,
        input_dilation_size = self.sizes[2],
        # number of layers
        n_layers = 5,
        p_stochastic = [0, 0, 0, 0, 0],
        #p_stochastic = [1.6/10, 1.4/10, 1.2/10, 1.0/10, 0.8/10],
)
        self.DiWTBlock4 = DiWTBlock(
        in_channels = basic_ch*8,
        out_channels = basic_ch*8,
        squeeze_ratio = 0.25,
        expansion_ratio = 4,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99),
        activation_layer=nn.GELU,
        head_dim = 32,
        mlp_ratio = 4,
        mlp_dropout = 0,
        attention_dropout = 0,
        # partitioning parameters
        partition_size = 8,
        input_dilation_size = self.sizes[3],
        # number of layers
        n_layers = 2,
        p_stochastic = [0, 0],
        #p_stochastic = [2.0/10, 1.8/10],
)
        self.up1 = up_conv(basic_ch*8,basic_ch*4)
        self.up2 = up_conv(basic_ch*4,basic_ch*2)
        self.up3 = up_conv(basic_ch*2,basic_ch)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
        self.weight3 = Scale(1)
        self.weight4 = Scale(1)
    def forward(self, x1, x2, x3, x4):
        x_out4 = self.DiWTBlock4(x4) +self.weight4(x4) #(B 128 256 256) (B 256 128 128)
        x4 = self.up1(x_out4)
        x3 = x4 + x3
        x_out3 = self.DiWTBlock3(x3) + self.weight3(x3)
        x3 = self.up2(x_out3)
        x2 = x3 + x2
        x_out2 = self.DiWTBlock2(x2) + self.weight2(x2)
        x2 = self.up3(x_out2)
        x1 = x2 + x1
        x_out1 = self.DiWTBlock1(x1) + self.weight1(x1)
        return x_out1
    
    

class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = ResNet(layers = [3, 4, 6, 3])
        self.groupconv1 = nn.Conv2d(64, 64, kernel_size=3, groups=32,padding=1)
        self.groupconv2 = nn.Conv2d(256, 128, kernel_size=3, groups=32,padding=1)
        self.groupconv3 = nn.Conv2d(512, 256, kernel_size=3, groups=32,padding=1)
        self.groupconv4 = nn.Conv2d(1024, 512, kernel_size=3, groups=32,padding=1)
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        x1, x2, x3, x4 = self.model(x)
        x1 = self.groupconv1(x1)
        x2 = self.groupconv2(x2)
        x3 = self.groupconv3(x3)
        x4 = self.groupconv4(x4)
        #(B,64,1024,1024) x3(B 256 256 256)
        return x1, x2, x3, x4






class  DiWTBR(nn.Module):
    def __init__(self, basic_ch=64):
        super( DiWTBR, self).__init__()
        self.basic_ch = basic_ch
        self.extract_feature = base_resnet()
        self.DiWTBU = DiWTBU(input_size = (384,512),basic_ch = self.basic_ch)
        self.Upsample = nn.Sequential(
            nn.Conv2d(basic_ch, 4 * basic_ch, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.defocus1 = nn.Conv2d(basic_ch, 3, kernel_size=3, stride=1, padding=1)
        self.sub_mean = MeanShift1(rgb_range = 1)
        self.add_mean = MeanShift2(rgb_range = 1, sign=1)

    def forward(self, image):
        image = self.sub_mean(image)
        F1, F2, F3, F4 = self.extract_feature(image)
        x_out1 = self.DiWTBU(F1, F2, F3, F4)#(B 256 256 256) (B 512 128 128)
        x_out1 = self.Upsample(x_out1)
        x_out1 = self.defocus1(x_out1)
        x_out1 = self.add_mean(x_out1)
        return x_out1