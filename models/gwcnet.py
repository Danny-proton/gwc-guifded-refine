from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from math import log
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
from scipy import sparse as sp

#########bts#######

def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        
        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels*2, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)
    

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                                 kernel_size=1, stride=1, padding=0),
                                                                       nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        
        return net

class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]
        
        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio
        
        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)

#########################

class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x=x.repeat(1,3,1,1)
        x = self.firstconv(x)
        #x[0,13,:,:]=0
        #print(x[:,14,:,:])
        l1 = self.layer1(x)
        
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        #l3[0,0,:,:]=0
        l4 = self.layer4(l3)      
        
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        fea_list=[x,l1,l2,l3,l4]
        # print(x.shape)
        # print(l1.shape)
        # print(l4.shape) 

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
           # print("!!!!!!!!!!")
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature,"feature":fea_list}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        # Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)


        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        
        
       

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class GwcNet(nn.Module):
    def __init__(self, args, use_concat_volume=True):
        super(GwcNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        

        for m in self.modules():
            # m.weight.requires_grad=False
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.weight.requires_grad=False
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.weight.requires_grad=False
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.requires_grad=True
                #m.bias.requires_grad=True
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                #m.weight.requires_grad=True
                #m.bias.requires_grad=True
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
               # m.bias.requires_grad=False
               
     #bts part
        feat_out_channels = [32, 32, 64, 128, 128]
        num_features = 192
     
        self.bn4_2      = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.daspp_3    = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6    = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12   = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18   = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24   = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.reduc8x8   = reduction_1x1(num_features // 4, num_features // 4, self.maxdisp)
        self.lpg8x8     = local_planar_guidance(8)
        
        self.upconv3    = upconv(num_features // 4, num_features // 4)
        self.bn4        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.bn3        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.bn2        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.bn1        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv1      = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[4] + 1, num_features, 3, 1, 1, bias=False),
                                            #   nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False)
                                              nn.ELU())
        self.upconv2    = upconv(num_features // 4, num_features // 8)
        self.conv2      = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[2] + 1, num_features , 3, 1, 1, bias=False),
                                              nn.ELU())
        
        self.conv3      = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[1] + 1, num_features , 3, 1, 1, bias=False),
                                              nn.ELU())
        
        self.conv4      = torch.nn.Sequential(nn.Conv2d(num_features + 5, num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        
        self.reduc4x4   = reduction_1x1(num_features // 4, num_features // 8, self.maxdisp)
        self.lpg4x4     = local_planar_guidance(4)
        
        self.reduc4x4_1   = reduction_1x1(num_features // 4, num_features // 8, self.maxdisp)
        self.lpg4x4_1     = local_planar_guidance(4)
        
        self.reduc4x4_2   = reduction_1x1(num_features // 4, num_features // 8, self.maxdisp)
        self.lpg4x4_2     = local_planar_guidance(4)
        
        self.reduc4x4_3   = reduction_1x1(num_features // 4, num_features // 8, self.maxdisp)
        self.lpg4x4_3     = local_planar_guidance(4)
        
        
        self.upconv1    = upconv(num_features // 8, num_features // 16)
        self.reduc1x1   = reduction_1x1(num_features, num_features // 32, self.maxdisp, is_final=True)
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())

    def forward(self, left, right):
        focal=0 # \?????
        #left_edge=y_gradient_1order(x_gradient_1order(left))
        #right_edge=y_gradient_1order(x_gradient_1order(right))
        #mask=left_edge>0.5
        #print(left,right)
        #right=right/2.0
        #pred_tra=mutual_info_pred(left,right,self.maxdisp)
        #print("left")
        features_left = self.feature_extraction(left)
        #print("right")
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        # add by yyx
        # "gwc_feature": gwc_feature, "concat_feature": concat_feature,"feature":fea_list
        # fea_list=[x,l1,l2,l3,l4]
        left_fea=features_left["feature"]
# fea_list=[x,l1,l2,l3,l4] torch.Size([1, 32, 256, 480])                                                           │amp: 1604494044.58921).
# fea_list=[x,l1,l2,l3,l4] torch.Size([1, 32, 256, 480])                                                           │
# fea_list=[x,l1,l2,l3,l4] torch.Size([1, 64, 128, 240])                                                           ├──────────────────────────────────────────────────────────
# fea_list=[x,l1,l2,l3,l4] torch.Size([1, 128, 128, 240])                                                          │Every 1.0s: gpustat -cpu          Sun Nov 15 13:05:32 2020
# fea_list=[x,l1,l2,l3,l4] torch.Size([1, 128, 128, 240])
        right_fea=features_right["feature"]
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        # print("cost0",cost0.shape)
        out1 = self.dres2(cost0)#to 1/8
        # print("out1",out1.shape)
        out2 = self.dres3(out1)#to 1/4
        # print("out2",out2.shape)
        out3 = self.dres4(out2)#to 1/2
        # print("out3",out3.shape)
#add bts
        # print("before classif",cost0)
        cost0 = self.classif0(cost0)
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2)
        cost3 = self.classif3(out3)
        
        cost0_14=cost0
        cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost0 = torch.squeeze(cost0, 1)
        
        cost1_14=cost1
        cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost1 = torch.squeeze(cost1, 1)
        
        cost2_14=cost2
        cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost2 = torch.squeeze(cost2, 1)
        
        cost3_14=cost3
        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        
        
        
    #for 8*8 to 4*4
        cost0_14 = torch.squeeze(cost0_14, 1)
        reduc4x4 = self.reduc4x4(cost0_14)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.maxdisp

        upconv1=cost0
        upconv1 = self.bn1(upconv1)
        group_size1=(upconv1.shape)[1]
        left_fea_low=F.upsample(left_fea[4], [left.size()[2], left.size()[3]], mode='nearest')
        concat1 = torch.cat([upconv1, left_fea_low, depth_4x4_scaled], dim=1)
        # upconv3, torch.Size([1, 128, 64, 128])
        # skip1, torch.Size([1, 96, 64, 128])
        # skip0[1, 96, 128, 256]
        # depth_8x8_scaled_ds, torch.Size([1, 1, 64, 128])
        # upconv1 torch.Size([1, 192, 256, 512])                                                                           │d by a TensorFlow restart. Purging 1000 expired tensor eve
        # target torch.Size([1, 192, 256, 512])                                                                            │nts from Tensorboard display between the previous step: 71
        # after change left_fea_low torch.Size([1, 128, 64, 128])                                                                       │ (timestamp: 1604482673.404424) and current step: 0 (times
        # after change2 depth_4x4_scaled torch.Size([1, 1, 256, 512])
        out_1= self.conv1(concat1)
        
    #for 4*4 to 2*2
        # print("it is here!!!!!!")
        cost1_14 = torch.squeeze(cost1_14, 1)
        reduc4x4_1 = self.reduc4x4_1(cost1_14)
        plane_normal_4x4_1 = reduc4x4_1[:, :3, :, :]
        plane_normal_4x4_1 = F.normalize(plane_normal_4x4_1, 2, 1)
        plane_dist_4x4_1 = reduc4x4_1[:, 3, :, :]
        plane_eq_4x4_1 = torch.cat([plane_normal_4x4_1, plane_dist_4x4_1.unsqueeze(1)], 1)
        depth_4x4_1 = self.lpg4x4_1(plane_eq_4x4_1, focal)
        depth_4x4_1_scaled = depth_4x4_1.unsqueeze(1) / self.maxdisp
            
        upconv2=cost1
        upconv2 = self.bn2(upconv2)
        group_size2=(upconv2.shape)[1]
        left_fea_middle=F.upsample(left_fea[2], [left.size()[2], left.size()[3]], mode='nearest')
        concat2 = torch.cat([upconv2, left_fea_middle, depth_4x4_1_scaled], dim=1)
        out_2= self.conv2(concat2)
    
    # 2
        cost2_14 = torch.squeeze(cost2_14, 1)
        reduc4x4_2 = self.reduc4x4_2(cost1_14)
        plane_normal_4x4_2 = reduc4x4_2[:, :3, :, :]
        plane_normal_4x4_2 = F.normalize(plane_normal_4x4_2, 2, 1)
        plane_dist_4x4_2 = reduc4x4_2[:, 3, :, :]
        plane_eq_4x4_2 = torch.cat([plane_normal_4x4_2, plane_dist_4x4_2.unsqueeze(1)], 1)
        depth_4x4_2 = self.lpg4x4_2(plane_eq_4x4_2, focal)
        depth_4x4_2_scaled = depth_4x4_2.unsqueeze(1) / self.maxdisp
            
        upconv3=cost2
        upconv3 = self.bn3(upconv3)
        group_size3=(upconv3.shape)[1]
        left_fea_high=F.upsample(left_fea[1], [left.size()[2], left.size()[3]], mode='nearest')
        concat3 = torch.cat([upconv3, left_fea_high, depth_4x4_2_scaled], dim=1)
        out_3= self.conv3(concat3)
        
    # 3
        cost3_14 = torch.squeeze(cost3_14, 1)
        reduc4x4_3 = self.reduc4x4_3(cost3_14)
        plane_normal_4x4_3 = reduc4x4_3[:, :3, :, :]
        plane_normal_4x4_3 = F.normalize(plane_normal_4x4_3, 2, 1)
        plane_dist_4x4_3 = reduc4x4_3[:, 3, :, :]
        plane_eq_4x4_3 = torch.cat([plane_normal_4x4_3, plane_dist_4x4_3.unsqueeze(1)], 1)
        depth_4x4_3 = self.lpg4x4_3(plane_eq_4x4_3, focal)
        depth_4x4_3_scaled = depth_4x4_3.unsqueeze(1) / self.maxdisp
        
        upconv4=cost3
        upconv4 = self.bn4(upconv4)
        reduc1x1 = self.reduc1x1(upconv4)
        concat3 = torch.cat([upconv3, reduc1x1, depth_4x4_scaled, depth_4x4_1_scaled,depth_4x4_2_scaled,depth_4x4_3_scaled], dim=1)
        out_4 = self.conv4(concat3)
        
    #for 2*2 to 1*1
        # cost2_cla = self.classif0(out2)
        # cost2_cla = torch.squeeze(cost2_cla, 1)
    
        # reduc2x2 = self.reduc2x2(cost2_14)
        # plane_normal_2x2 = reduc2x2[:, :3, :, :]
        # plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
        # plane_dist_2x2 = reduc2x2[:, 3, :, :]
        # plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        # depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        # depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.maxdisp
        
        # upconv3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        # group_size3=(upconv3.shape)[1]
        # reduc1x1 = self.reduc1x1(upconv3)
        # concat3 = torch.cat([upconv3, reduc1x1, depth_2x2_scaled.unsqueeze(1).expand(-1,group_size3,-1,-1,-1), depth_4x4_1_scaled.unsqueeze(1).expand(-1,group_size3,-1,-1,-1), depth_8x8_scaled.unsqueeze(1).expand(-1,group_size3,-1,-1,-1)], dim=2)
        # out3 = self.conv1(concat3)

        if self.training:
        #if True:

            # print("after squeeze",cost0.shape)
            pred0 = F.softmax(out_1, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            pred1 = F.softmax(out_2, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            pred2 = F.softmax(out_3, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            pred3 = F.softmax(out_4, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]

        else:

            cost3 = torch.squeeze(out_4, 1)
            pred3 = F.softmax(cost3, dim=1)
            # _,d,H,W=pred3.size()
            # for h in range(0,H,10):
            #     for w in range(0,W,20):
            #         point_disp=pred3[0,:,h,w]
            #         distribution=point_disp.view(-1)
            #         plt.xlabel('Disparity')
            #         plt.ylabel('Probability')
            #         x=np.arange(len(distribution))
            #         print(h+1,"_",w+1,distribution)
            #         plt.xlim(xmax=192, xmin=0)
            #         plt.ylim(ymax=1,ymin=0)
                    
            #         #plt.hist(x=distribution, bins=d, color='#0504aa',alpha=0.7, rwidth=0.9)
            #         plt.bar(x, distribution,fc='r')
            #         filename='fea_map/kitti/disparity_distribution_4/'+str(h+1)+'_'+str(w+1)+'.png'
            #         plt.savefig(filename)
            #         plt.close()
            confidence,index=torch.max(pred3,dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3],confidence,index
            #return pred_tra


def GwcNet_G(d):
    return GwcNet(d, use_concat_volume=False)


def GwcNet_GC(d):
    return GwcNet(d, use_concat_volume=True)



# def entropy(labels):
#
#     if len(labels) == 0:
#         return 1.0
#     label_idx = np.unique(labels, return_inverse=True)[1]
#     pi = np.bincount(label_idx).astype(np.float64)
#     #print("!!!",pi)
#     #pi = pi[pi > 0]
#     pi_sum = np.sum(pi)
#     # log(a / b) should be calculated as log(a) - log(b) for
#     # possible loss of precision
#     entro=-((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
#     entro = np.where(np.isnan(entro), 0, entro)
#     return entro
def entropy(pi):
    #pi = pi[pi >= 0]
    #print(pi)
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    join_entropy=-((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
    #print(join_entropy)
    #zero=np.zeros([1,6,6])
    join_entropy=np.where(np.isnan(join_entropy),0,join_entropy)
    #join_entropy[0, 0] = 0
    return join_entropy
def mutual_entropy(a,b,bin_num):
    a=np.array(a)
    b=np.array(b)
    #print(a.shape)
    a_hist=np.histogram(a,bins=bin_num+1,range=(0,bin_num))[0]
    b_hist = np.histogram(b, bins=bin_num+1, range=(0, bin_num))[0]
    #print("!!!!!",a.shape,a_hist.shape)
    #a = a.reshape(1,-1)
    #b = b.reshape(1,-1)
    # print(a.ravel().shape,b.ravel().shape)
    ab_hist=np.histogram2d(a.ravel(),b.ravel(),bins=(bin_num+1,bin_num+1),range=[(0,bin_num),(0,bin_num)])[0]
    # print(ab_hist.shape)
    # print("a entropy",entropy(a_hist))
    # print("b entropy",entropy(b_hist))
    # print("ab entropy",entropy(ab_hist)[7])
    # a_entropy=np.repeat(entropy(a_hist),[bin_num],axis=0)
    # b_entropy = np.repeat(entropy(b_hist),bin_num,axis=0)
    # print(a_entropy.shape)
    # mutual_entro=np.zeros([bin_num,bin_num])
    # for b in range(bin_num):
    #     mutual_entro[b]=entropy(a_hist)+entropy(b_hist)-(entropy(ab_hist))[b]
    mutual_entro=[entropy(a_hist),entropy(b_hist),entropy(ab_hist)]
    #print("!!",mutual_entro[0].shape)
    return mutual_entro

def mutual_info_pred(left, right,maxdisp):
    # left=(left+0.5).floor()
    # right=(right+0.5).floor()
    #print(left,right)
    mutual_info=mutual_entropy(left,right,255)
    B, C, H, W = left.shape
    volume = left.new_zeros([B, maxdisp, H, W])
    for x in range(H):
        for y in range(W):
            for i in range(maxdisp):
                if x<=i:
                    volume[:,i,x,y]=0
                else:
                    left_i=int(left[0,0,x,y])
                    right_i=int(right[0,0,x-i,y])
                    # print(left_i,right_i)
                    #print(mutual_info[0].shape)
                    volume[0,i,x,y]=mutual_info[0][left_i]+mutual_info[1][right_i]-mutual_info[2][left_i,right_i]
                #print(volume[0,i,x,y])
    volume=torch.softmax(volume,dim=1)
    confidence,pred=torch.max(volume,dim=1)
    pred=pred.float()
    pred_sub=disparity_regression(volume,maxdisp)
    return pred

def x_gradient_1order(img):
    img = img.permute(0,2,3,1)
    img_l = img[:,:,1:,:] - img[:,:,:-1,:]
    img_r = img[:,:,-1,:] - img[:,:,-2,:]
    img_r = img_r.unsqueeze(2)
    img  = torch.cat([img_l, img_r], 2).permute(0, 3, 1, 2)
    return img

def y_gradient_1order(img):
    # pdb.set_trace()
    img = img.permute(0,2,3,1)
    img_u = img[:,1:,:,:] - img[:,:-1,:,:]
    img_d = img[:,-1,:,:] - img[:,-2,:,:]
    img_d = img_d.unsqueeze(1)
    img  = torch.cat([img_u, img_d], 1).permute(0, 3, 1, 2)
    return img

def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad
