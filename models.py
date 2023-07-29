import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.mSwin import * #mSwin, mViT
from basic import *
import torchvision
import matplotlib.pyplot as plt
# import imagenet

class MobileNetV1(nn.Module):
    def __init__(self, relu6=True, modality='rgb'):
        super(MobileNetV1, self).__init__()
        assert modality in ['rgb', 'depth']
        self.modality = modality
        
        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6))

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6))

        if self.modality == 'rgb':
            input_layer = conv_bn(3, 32, 2, relu6)
        elif self.modality == 'depth':
            input_layer = conv_bn(1, 32, 2, relu6)
        else:
            raise ValueError('MobileNet V1 modality must be ["rgb", "depth"].')

        self.layers = nn.Sequential(
            input_layer, 
            conv_dw( 32,  64, 1, relu6),
            conv_dw( 64, 128, 2, relu6),
            conv_dw(128, 128, 1, relu6),
            conv_dw(128, 256, 2, relu6),
            conv_dw(256, 256, 1, relu6),
            conv_dw(256, 512, 2, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 1024, 2, relu6),
            conv_dw(1024, 1024, 1, relu6),
        )
        weights_init(self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class MobileNetV1_simplified(nn.Module):
    def __init__(self, relu6=True, modality='depth'):
        super(MobileNetV1_simplified, self).__init__()
        assert modality in ['rgb', 'depth']
        self.modality = modality
        self.mode = modality
        
        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6))

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6))

        if self.modality == 'rgb':
            input_layer = conv_bn(3, 32, 2, relu6)
        elif self.modality == 'depth':
            input_layer = conv_bn(1, 32, 2, relu6)
        else:
            raise ValueError('MobileNet V1 modality must be ["rgb", "depth"].')

        self.layers = nn.Sequential(
            input_layer, 
            conv_dw( 32,   64, 1, relu6),
            conv_dw( 64,  128, 2, relu6),
            conv_dw(128,  256, 2, relu6),
            conv_dw(256,  512, 2, relu6),
            conv_dw(512, 1024, 2, relu6),
        )
        weights_init(self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class SeedBinRegressor(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):  
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0), 
            nn.GELU(),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self,x):
        """
        Input shape (N, E, H, W)
        """
        B = self._net(x)
        B = self.pool(B)
        eps = 1e-3
        B = B + eps
        B_widths_normed = B / B.sum(dim=1, keepdim=True)
        B_widths = (self.max_depth - self.min_depth) * B_widths_normed  # .shape NCHW
        B_widths_normed = B_widths_normed[:, :, 0, 0]
        # pad has the form (left, right, top, bottom, front, back)
        # B_widths = nn.functional.pad(B_widths, (0,0,0,0,0,1), mode='constant', value=self.min_depth)
        B_widths = nn.functional.pad(B_widths, (1, 0), mode='constant', value=self.min_depth)
        # B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape N, B, 1, 1
        B_edges = B_edges[:, :, 0, 0]
        # centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        B_centers = 0.5 * (B_edges[:, :-1] + B_edges[:,1:])

        # output shape: (N, b)
        return B_widths_normed, B_edges, B_centers

class SeedBinRegressor_DR(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):  
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0), 
            nn.GELU(),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.ReLU()
        )
        self._bias_net = nn.Sequential(
            nn.Conv2d(in_features, 32, 1, 1, 0), 
            nn.GELU(),
            nn.Conv2d(32, 2, 1, 1, 0),
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self,x, min_val, max_val):
        """
        Input shape (N, E, H, W)
        """
        B = self._net(x)
        B = self.pool(B)
        eps = 1e-3
        B = B + eps
        B_widths_normed = B / B.sum(dim=1, keepdim=True)
        B_widths_normed = B_widths_normed[:, :, 0, 0]
        bias = self._bias_net(x)
        bias = bias[:, :, 0, 0]
        d_min, d_max = torch.chunk(bias, 2, dim=1)
        min = min_val - d_min
        min[min < self.min_depth] = self.min_depth
        max = max_val + d_max
        max[max > self.max_depth] = self.max_depth
        diff =  max - min 
        B_widths = diff*B_widths_normed
        B_widths = torch.cat((min, B_widths), dim=1)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape N, B, 1, 1

        B_centers = 0.5 * (B_edges[:, :-1] + B_edges[:,1:])

        # output shape: (N, b)
        return B_widths_normed, B_edges, B_centers

class SigmoidSplitter(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, prev_nbins, 1, 1, 0),
            nn.AdaptiveAvgPool2d((1, 1))
            # nn.ReLU()
        )
    
    def forward(self, x, b_prev, prev_b_embedding=None):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins
        """
        if prev_b_embedding is not None:
            prev_b_embedding = F.interpolate(prev_b_embedding, scale_factor=2, mode='nearest')
            x = x + prev_b_embedding
        x = self._net(x)    #.shape N, b, 1, 1
        x = x.squeeze(3)
        S = torch.sigmoid(x)  # .shape n,c, 1; 0<S<1
        # print(S.size())
        n, c, _ = S.shape
        # S = S.unsqueeze(2)  # .shape n, c, 1, h, w
        S_normed = torch.cat((S, 1-S), dim=2)  # fractional splits , .shape n, prev_nbins, 2

        # b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees, .shape N, prev_bins
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1,2).view(n, 2*c)  # .shape n, prev_nbins * split_factor, h, w

        # calculate bin centers for loss calculation
        B_widths = (self.max_depth - self.min_depth) * b  # .shape N, nprev * splitfactor, H, W
        # pad has the form (left, right, top, bottom, front, back)
        # B_widths = nn.functional.pad(B_widths, (0,0,0,0,0,1), mode='constant', value=self.min_depth)
        B_widths = nn.functional.pad(B_widths, (1, 0), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW
        # print(B_edges.size())

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        # print(B_widths.size())
        return b, B_edges, B_centers

class SigmoidSplitter_DR(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            # nn.BatchNorm2d(mlp_dim),
            nn.GELU(),
            nn.Conv2d(mlp_dim, prev_nbins, 1, 1, 0),
            # nn.BatchNorm2d(prev_nbins),
            nn.AdaptiveAvgPool2d((1, 1))
            # nn.ReLU()
        )
        self._bias_net = nn.Sequential(
            nn.Conv2d(in_features, 32, 1, 1, 0), 
            nn.GELU(),
            nn.Conv2d(32, 2, 1, 1, 0),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x, b_prev, prev_b_embedding, min_val, max_val):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins
        """
        if prev_b_embedding is not None:
            prev_b_embedding = F.interpolate(prev_b_embedding, scale_factor=2, mode='nearest')
            x = x + prev_b_embedding
        bias = self._bias_net(x)
        x = self._net(x)    #.shape N, b, 1, 1
        x = x.squeeze(3)
        S = torch.sigmoid(x)  # .shape n,c, 1; 0<S<1
        # print(S.size())
        n, c, _ = S.shape
        # S = S.unsqueeze(2)  # .shape n, c, 1, h, w
        S_normed = torch.cat((S, 1-S), dim=2)  # fractional splits , .shape n, prev_nbins, 2

        # b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees, .shape N, prev_bins
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1,2).view(n, 2*c)  # .shape n, prev_nbins * split_factor, h, w
        # print(b.size())
        bias = bias[:, :, 0, 0]
        d_min, d_max = torch.chunk(bias, 2, dim=1)

        min = min_val + d_min
        min[min < self.min_depth] = self.min_depth
        max = max_val + d_max
        max[max > self.max_depth] = self.max_depth

        diff = max - min
        # print(B_edges.size())
        # print('-')
        # print('min:{}, max:{}'.format(min[0], max[0]))
        B_widths = diff * b  # .shape N, nprev * splitfactor, H, W
        B_widths = torch.cat((min, B_widths), dim=1)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        # print(B_widths.size())
        return b, B_edges, B_centers

class Projector(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0),
        )

    def forward(self, x):
        return self._net(x)

class Bins_predictor(nn.Module):
    def __init__(self, init_bins=4, embedding_dim=32, hidden_dim=64, min_val=1e-3, max_val=10):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    
        self.bins_emb0 = Projector(in_features=512, out_features=embedding_dim, mlp_dim=hidden_dim)
        self.bins_emb1 = Projector(in_features=256, out_features=embedding_dim, mlp_dim=hidden_dim)
        self.bins_emb2 = Projector(in_features=128, out_features=embedding_dim, mlp_dim=hidden_dim)
        self.bins_emb3 = Projector(in_features=64, out_features=embedding_dim, mlp_dim=hidden_dim)

        self.bins_seed_mlp = SeedBinRegressor(in_features=embedding_dim, n_bins=init_bins, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)
        self.split1 = SigmoidSplitter(in_features=embedding_dim, prev_nbins=init_bins, split_factor=2, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)
        self.split2 = SigmoidSplitter(in_features=embedding_dim, prev_nbins=2*init_bins, split_factor=2, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)
        self.split3 = SigmoidSplitter(in_features=embedding_dim, prev_nbins=4*init_bins, split_factor=2, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)

        self.conv_out = nn.Sequential(nn.Conv2d(32, 8*init_bins, 3, 1, 1), nn.Softmax(dim=1))

        weights_init(self)

    def forward(self, features):
        """
        # Input depth: 1, h, w
        # Input guide: 32, h, w
        """
        c0 = features[0] # 32
        c1 = features[1] # 64
        c2 = features[2] # 128
        c3 = features[3] # 256
        c4 = features[4] # 512
        # Bins branch
        e0 = self.bins_emb0(c0)
        e1 = self.bins_emb1(c1)
        e2 = self.bins_emb2(c2)
        e3 = self.bins_emb3(c3)
        
        b0, B0_edges, b0_centers = self.bins_seed_mlp(e0)
        b1, B1_edges, b1_centers = self.split1(e1, b0, e0)
        b2, B2_edges, b2_centers = self.split2(e2, b1, e1)
        b3, B3_edges, b3_centers = self.split3(e3, b2, e2)

        n, dout = b3_centers.size()
        b3_centers = b3_centers.view(n, dout, 1, 1)

        bin_depth = self.conv_out(c4)
        bin_depth = torch.sum(bin_depth * b3_centers, dim=1, keepdim=True)
        bin_edges = [B0_edges, B1_edges, B2_edges, B3_edges]
        return bin_depth, bin_edges

class Bins_predictor_DR(nn.Module):
    def __init__(self, init_bins=4, embedding_dim=32, hidden_dim=64, min_val=1e-3, max_val=10):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    
        self.bins_emb0 = Projector(in_features=512, out_features=embedding_dim, mlp_dim=hidden_dim)
        self.bins_emb1 = Projector(in_features=256, out_features=embedding_dim, mlp_dim=hidden_dim)
        self.bins_emb2 = Projector(in_features=128, out_features=embedding_dim, mlp_dim=hidden_dim)
        self.bins_emb3 = Projector(in_features=64, out_features=embedding_dim, mlp_dim=hidden_dim)

        self.bins_seed_mlp = SeedBinRegressor_DR(in_features=embedding_dim, n_bins=init_bins, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)
        self.split1 = SigmoidSplitter_DR(in_features=embedding_dim, prev_nbins=init_bins, split_factor=2, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)
        self.split2 = SigmoidSplitter_DR(in_features=embedding_dim, prev_nbins=2*init_bins, split_factor=2, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)
        self.split3 = SigmoidSplitter_DR(in_features=embedding_dim, prev_nbins=4*init_bins, split_factor=2, mlp_dim=hidden_dim, min_depth=self.min_val, max_depth=self.max_val)

        self.conv_out = nn.Sequential(nn.Conv2d(32, 8*init_bins, 3, 1, 1), nn.Softmax(dim=1))

        weights_init(self)

    def forward(self, features, input_depth):
        """
        # Input depth: 1, h, w
        # Input guide: 32, h, w
        """

        n, _, h, w = input_depth.size()
        input_depth = input_depth.view(n, h*w)

        min_val = torch.min(input_depth, dim=1, keepdim=True)[0]
        max_val = torch.max(input_depth, dim=1, keepdim=True)[0]

        c0 = features[0] # 32
        c1 = features[1] # 64
        c2 = features[2] # 128
        c3 = features[3] # 256
        c4 = features[4] # 512
        # Bins branch
        e0 = self.bins_emb0(c0)
        e1 = self.bins_emb1(c1)
        e2 = self.bins_emb2(c2)
        e3 = self.bins_emb3(c3)
        
        b0, B0_edges, b0_centers = self.bins_seed_mlp(e0, min_val, max_val)
        b1, B1_edges, b1_centers = self.split1(e1, b0, e0, min_val, max_val)
        b2, B2_edges, b2_centers = self.split2(e2, b1, e1, min_val, max_val)
        b3, B3_edges, b3_centers = self.split3(e3, b2, e2, min_val, max_val)
        n, dout = b3_centers.size()
        b3_centers = b3_centers.view(n, dout, 1, 1)

        bin_depth = self.conv_out(c4)
        bin_depth = torch.sum(bin_depth * b3_centers, dim=1, keepdim=True)
        bin_edges = [B0_edges, B1_edges, B2_edges, B3_edges]
        return bin_depth, bin_edges

########################################################################################################
class MobileUent_simpleUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'MobileUnet'
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        rgbnet = MobileNetV1_simplified(modality='rgb')
        deptdepthet = MobileNetV1_simplified(modality='depth')
        for i in range(6):
            setattr(self, 'rgb_conv{}'.format(i), rgbnet.layers[i])
            setattr(self, 'depth_conv{}'.format(i), deptdepthet.layers[i])

        kernel_size = 5
        self.decode_conv1 = simpleUp(1024, 512, k=kernel_size)
        self.decode_conv2 = simpleUp(512, 256, k=kernel_size)
        self.decode_conv3 = simpleUp(256, 128, k=kernel_size)
        self.decode_conv4 = simpleUp(128, 64, k=kernel_size)
        self.decode_conv5 = simpleUp(64, 32, k=kernel_size)
        self.conv = nn.Conv2d(32, 1, 1, 1, 0)

        weights_init(self)

    def forward(self, rgb, depth):

        for i in range(6):
            rgb_layer = getattr(self, 'rgb_conv{}'.format(i))
            rgb = rgb_layer(rgb)
            depth_layer = getattr(self, 'depth_conv{}'.format(i))
            depth = depth_layer(depth)

            if i==1:
                r1 = rgb 
                d1 = depth
            elif i==2:
                r2 = rgb 
                d2 = depth
            elif i==3:
                r3 = rgb 
                d3 = depth
            elif i==4:
                r4 = rgb 
                d4 = depth

        features = []
        for i in range(1, 6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            if i == 5:
                x0 = layer(r1, d1, x0)
                features.append(x0)
            elif i==4:
                x0 = layer(r2, d2, x0)
                features.append(x0)
            elif i==3:
                x0 = layer(r3, d3, x0)
                features.append(x0)
            elif i==2:
                x0 = layer(r4, d4, x0)
                features.append(x0)
            elif i==1:
                x0 = layer(rgb, depth, None)
                features.append(x0)

        d = self.conv(x0)
        return d, features 

class MobileUent_UpCSPN(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.name = 'MobileUnet_UpCSPN{}'.format(k)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        rgbnet = MobileNetV1_simplified(modality='rgb')
        deptdepthet = MobileNetV1_simplified(modality='depth')
        for i in range(6):
            setattr(self, 'rgb_conv{}'.format(i), rgbnet.layers[i])
            setattr(self, 'depth_conv{}'.format(i), deptdepthet.layers[i])

        kernel_size = 5
        self.decode_conv1 = UpCSPN(1024, 512, k)
        self.decode_conv2 = UpCSPN(512, 256, k)
        self.decode_conv3 = UpCSPN(256, 128, k)
        self.decode_conv4 = UpCSPN(128, 64, k)
        self.decode_conv5 = UpCSPN(64, 32, k)
        self.conv = nn.Conv2d(32, 1, 1, 1, 0)

        weights_init(self)

    def forward(self, rgb, depth):

        for i in range(6):
            rgb_layer = getattr(self, 'rgb_conv{}'.format(i))
            rgb = rgb_layer(rgb)
            depth_layer = getattr(self, 'depth_conv{}'.format(i))
            depth = depth_layer(depth)

            if i==1:
                r1 = rgb 
                d1 = depth
            elif i==2:
                r2 = rgb 
                d2 = depth
            elif i==3:
                r3 = rgb 
                d3 = depth
            elif i==4:
                r4 = rgb 
                d4 = depth

        features = []
        for i in range(1, 6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            if i == 5:
                x0 = layer(r1, d1, x0)
                features.append(x0)
            elif i==4:
                x0 = layer(r2, d2, x0)
                features.append(x0)
            elif i==3:
                x0 = layer(r3, d3, x0)
                features.append(x0)
            elif i==2:
                x0 = layer(r4, d4, x0)
                features.append(x0)
            elif i==1:
                x0 = layer(rgb, depth, 0)
                features.append(x0)

        d = self.conv(x0)
        return d, features 

########################################################################################################
class AutoEncoder_simpleUp(nn.Module):
    def __init__(self, lowRes=True):
        super().__init__()
        self.name = 'AutoEncoder_simpleUp'
        self.lowRes = lowRes
        self.autoencoder = MobileUent_simpleUp()

    def forward(self, rgb, depth):
        depth = depth * 10
        if self.lowRes:
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        d, features = self.autoencoder(rgb, depth)
        return d, features

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AutoEncoder_UpCSPN(nn.Module):
    def __init__(self, lowRes=True, k=3):
        super().__init__()
        self.name = 'AutoEncoder_UpCSPN{}'.format(k)
        self.lowRes = lowRes
        self.autoencoder = MobileUent_UpCSPN(k)

    def forward(self, rgb, depth):
        depth = depth * 10
        if self.lowRes:
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        d, features = self.autoencoder(rgb, depth)
        return d, features

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AutoEncoder_simpleUp_bins(nn.Module):
    def __init__(self, lowRes=True, init_bins=4, embedding_dim=32, hidden_dim=64, min_val=1e-3, max_val=10):
        super().__init__()
        self.name = 'AutoEncoder_simpleUp_{}bins'.format(8*init_bins)
        self.lowRes = lowRes
        self.autoencoder = MobileUent_simpleUp()
        self.bins_layer = Bins_predictor(init_bins=init_bins,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                min_val=min_val,
                max_val=max_val)

    def forward(self, rgb, depth):
        depth = depth*10
        if self.lowRes:
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        # print(rgb.size())
        # print(depth.size())
        d, features = self.autoencoder(rgb, depth)
        bin_depth, bin_edges = self.bins_layer(features)
        return bin_edges, bin_depth

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AutoEncoder_UpCSPN_bins(nn.Module):
    def __init__(self, lowRes=True, k=3, init_bins=4, embedding_dim=32, hidden_dim=64, min_val=1e-3, max_val=10):
        super().__init__()
        self.name = 'AutoEncoder_UpCSPN{}_{}bins'.format(k, 8*init_bins)
        self.lowRes = lowRes
        self.autoencoder = MobileUent_UpCSPN(k)
        self.bins_layer = Bins_predictor(init_bins=init_bins,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                min_val=min_val,
                max_val=max_val)

    def forward(self, rgb, depth):
        if self.lowRes:
            depth = depth*10
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        d, features = self.autoencoder(rgb, depth)
        bin_depth, bin_edges = self.bins_layer(features)
        return bin_edges, bin_depth

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AutoEncoder_simpleUp_bins_DR(nn.Module):
    def __init__(self, lowRes=True, init_bins=4, embedding_dim=32, hidden_dim=64, min_val=1e-3, max_val=10):
        super().__init__()
        self.name = 'AutoEncoder_simpleUp_{}bins_DR'.format(8*init_bins)
        self.lowRes = lowRes
        self.autoencoder = MobileUent_simpleUp()
        self.bins_layer = Bins_predictor_DR(
                init_bins=init_bins,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                min_val=min_val,
                max_val=max_val)

    def forward(self, rgb, depth):
        # depth = depth * 10
        if self.lowRes:
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        # print(rgb.size())
        # print(depth.size())
        d, features = self.autoencoder(rgb, depth)
        bin_depth, bin_edges = self.bins_layer(features, depth)
        return bin_edges, bin_depth

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AutoEncoder_UpCSPN_bins_DR(nn.Module):
    def __init__(self, lowRes=True, k=3, init_bins=4, embedding_dim=32, hidden_dim=64, min_val=1e-3, max_val=10):
        super().__init__()
        self.name = 'AutoEncoder_UpCSPN{}_{}bins_DR'.format(k, 8*init_bins)
        self.lowRes = lowRes
        self.autoencoder = MobileUent_UpCSPN(k)
        self.bins_layer = Bins_predictor_DR(init_bins=init_bins,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                min_val=min_val,
                max_val=max_val)

    def forward(self, rgb, depth):
        if self.lowRes:
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        d, features = self.autoencoder(rgb, depth)
        bin_depth, bin_edges = self.bins_layer(features, depth)
        return bin_edges, bin_depth

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Autoencoder_simpleUp_AdaBins(nn.Module):
    def __init__(self, lowRes=True, n_bins=256, min_val=1e-3, max_val=10, norm='linear', hidden_dim=128):
        super().__init__()
        self.name = 'Autoencoder_simpleUp_AdaBins_{}bins'.format(n_bins)
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.lowRes = lowRes

        self.model = MobileUent_simpleUp()

        self.bins_layer = mViT(in_channels=32, 
                                  dim_out=n_bins, 
                                  n_query_channels=hidden_dim, 
                                  patch_size=16, 
                                  embedding_dim=hidden_dim, 
                                  num_heads=4)

        # self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        self.conv_R = nn.Sequential(nn.Conv2d(hidden_dim, n_bins, kernel_size=1, stride=1, padding=0), nn.Softmax(dim=1))

    def forward(self, rgb, depth):
        if self.lowRes:
            depth = depth*10
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        d, features = self.model(rgb, depth)
        bin_widths_normed, range_attention_maps = self.bins_layer(features[4])
        out = self.conv_R(range_attention_maps)

        B_edges = []
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        B_edges.append(bin_edges)

        return B_edges, pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Autoencoder_UpCSPN_AdaBins(nn.Module):
    def __init__(self, lowRes=True, k=3, n_bins=256, min_val=1e-3, max_val=10, norm='linear', hidden_dim=128):
        super().__init__()
        self.name = 'Autoencoder_UpCSPN{}_AdaBins_{}bins'.format(k, n_bins)
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.lowRes = lowRes

        self.autoencoder = MobileUent_UpCSPN(k)

        self.bins_layer = mViT(in_channels=32, 
                                  dim_out=n_bins, 
                                  n_query_channels=hidden_dim, 
                                  patch_size=16, 
                                  embedding_dim=hidden_dim, 
                                  num_heads=4)

        # self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        self.conv_R = nn.Sequential(nn.Conv2d(hidden_dim, n_bins, kernel_size=1, stride=1, padding=0), nn.Softmax(dim=1))

    def forward(self, rgb, depth):
        if self.lowRes:
            depth = depth*10
            depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)
        d, features = self.autoencoder(rgb, depth)
        bin_widths_normed, range_attention_maps = self.bins_layer(features[4])
        out = self.conv_R(range_attention_maps)

        B_edges = []
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        B_edges.append(bin_edges)

        return B_edges, pred