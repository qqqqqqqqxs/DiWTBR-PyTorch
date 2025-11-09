import torch

from torchvision import models
import torch.nn as nn
import random

class Weight_loss(nn.Module):
    def __init__(self):
        super(Weight_loss, self).__init__()

    def forward(self, image, est, target):
        weight = est
        diff = torch.pow(target-image, 2)
        loss = diff.mul(weight)

        loss = torch.mean(loss)
        return loss

class Normalize_loss(nn.Module):
    def __init__(self):
        super(Normalize_loss, self).__init__()

    def forward(self, image, target):
        N = image.size(0)
        diff = torch.pow(torch.abs(target-image), 1.5+1)
        norm = torch.pow(torch.abs(target-image), 1.5)
        loss = torch.Tensor([0.]).cuda()
        for i in range(N):
            up = torch.sum(diff[i])
            loss += up.div(torch.sum(norm[i]))
        loss = loss/N

        return loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        blocks = []
        #vgg16 = models.vgg16(pretrained=True)
        blocks.append(vgg16.features[:4].eval())
        blocks.append(vgg16.features[4:9].eval())
        #blocks.append(vgg16.features[9:16].eval())
        #blocks.append(vgg16.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        #self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[1], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            pos = random.randint(0, 1024-512)
            input = input[:, :, pos:pos+512, pos:pos+512]
            target = target[:, :, pos:pos+512, pos:pos+512]
            #input = self.transform(input, mode='bilinear', size=(512, 512), align_corners=False)
            #target = self.transform(target, mode='bilinear', size=(512, 512), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
                #print(loss)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.mse_loss(gram_x, gram_y)
                #print(loss)

        return loss