from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

def convdn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    #print(out_channels)
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         DomainNorm(out_channels))

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_r2l_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    # for i in range(disp):
    #     if i > 0:
    #         cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]

    #     else:
    #         cost[:, :, i, :, :] = refimg_feature - targetimg_feature
    # for i in range(disp):
    #     if i > 0:
    #         cost[:, :, i, :, :-i] = refimg_feature[:, :, :, :-i] - targetimg_feature[:, :, :, i:]

    #     else:
    #         cost[:, :, i, :, :] = refimg_feature - targetimg_feature
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, :-i] = groupwise_correlation(refimg_fea[:, :, :, :-i], targetimg_fea[:, :, :, i:],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


# class ChannelNorm(Module):
#     def __init__(self, eps=1e-5):
#         super(ChannelNorm, self).__init__()
# #        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
# #        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
# #        self.num_groups = num_groups
#         self.eps = eps

#     def forward(self, x):
# #        N,C,H,W = x.size()
# #        G = self.num_groups

# #        x = x.view(N,G,-1)
#         mean = x.mean(1, keepdim=True)
#         var = x.var(1, keepdim=True)

#         x = (x-mean) / (var+self.eps).sqrt()
#         return x
# #        x = x.view(N,C,H,W)
# #        return x * self.weight + self.bias

class DomainNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(DomainNorm, self).__init__()
        self.num_features=num_features
        self.eps=eps
        self.weight = nn.Parameter(torch.ones(1,self.num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,self.num_features,1,1))
        self.inbn=nn.InstanceNorm2d(self.num_features)

    def forward(self, x):
        # print("start")
        # print(x[0,:,0,1])
        x=self.inbn(x)
        # print(x[0,:,0,1])
        # print((torch.sum(torch.pow(x[0,:,0,1],2),dim=0)+self.eps).sqrt())
        #print(x.shape)
        #print(x.type)
        # mean = x.mean(1, keepdim=True)
        # var = x.var(1, keepdim=True)
        l2= torch.sum(torch.pow(x,2),dim=1)
        x = x / (l2+self.eps).sqrt()
        #print(x[0,:,0,1])
        #print(x.type)
        return x
        # return F.instance_norm(
        #     input, self.running_mean, self.running_var, self.weight, self.bias,
        #     self.training or not self.track_running_stats, self.momentum, self.eps)
        
