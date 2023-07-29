import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
import torchmetrics.functional as mF
import matplotlib.pyplot as plt

class LossFunctions():
    def scale_invariant_loss(self, scale=0.85):
        return SILogLoss(scale=scale)

    def l1_loss(self):
        return L1Loss()

    def l2_loss(self):
        return MaskedMSELoss()

    def gradient_loss(self):
        return GradientLoss()

    def bin_chamfer_loss(self):
        return BinsChamferLoss()

    def ssim_loss(self):
        return SSIMLoss()

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'L1Loss'
    def forward(self, outputs, target):
        mask = ((target > 0) * (outputs > 0)) > 0
        outputs = outputs[mask]
        target = target[mask]

        d = torch.abs(outputs-target).mean()
        return d

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self, scale=0.85):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.scale = scale

    def forward(self, outputs, target):
        mask = (((target > 0) * (outputs > 0)) > 0).detach()

        target = target[mask]
        outputs = outputs[mask]

        d = torch.log(outputs) - torch.log(target)
        D = torch.var(d) + (1-self.scale) * torch.pow(torch.mean(d), 2)
        loss = torch.sqrt(D)

        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target):
        mask = (((target>0) * (outputs>0)) > 0).detach()
        target = target[mask]
        outputs = outputs[mask]
        # masked_target.data.masked_fill_(~mask, float(100))
        # masked_input.data.masked_fill_(~mask, float(100))

        # masked_target = torch.log(masked_target)
        # masked_input = torch.log(masked_input)

        d = target - outputs
        dy, dx = mF.image_gradients(d)
        d_grad = torch.pow(dx, 2) + torch.pow(dy, 2)

        d_grad = d_grad[mask]

        loss = torch.mean(d_grad)

        return loss

# SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target):
        loss = 0.5*(1-ssim(outputs, target))
        return loss


class L1_SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, window_size=11, size_average = True):
        # l1 = torch.mean(torch.abs(target - outputs))
        l1 = nn.L1Loss() 
        l1_loss = l1(outputs, target)
        ssim_loss = 0.5*(1-ssim(outputs, target))
        return 0.85*ssim_loss + 0.15*l1_loss

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self, multi=True):
        super().__init__()
        self.name = "ChamferLoss"
        self.multi = multi

    def forward(self, bins, target_depth_maps):
        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = (target_points.ge(1e-3)).detach()  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1
        if self.multi==True:
            # input_points  []
            loss = 0
            for bin_edges in bins:
                bin_center = 0.5 * (bin_edges[:, 1:] + bin_edges[:, :-1])
                n, p = bin_center.shape
                input_points = bin_center.view(n, p, 1)
                # input_points.append(input_point)
                bin_loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
                loss += bin_loss
        else:
            bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
            n, p = bin_centers.shape
            input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
            loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        # n, c, h, w = target_depth_maps.shape

        # target_points = target_depth_maps.flatten(1)  # n, hwc
        # mask = target_points.ge(1e-3)  # only valid ground truth points
        # target_points = [p[m] for p, m in zip(target_points, mask)]
        # target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        # target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        return loss

class BinsChamferLoss_multi(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss_multi"
        self.partition = nn.Unfold(kernel_size=112, stride=112)

    def forward(self, bins, target_depth_maps):
        N, B, H, W = bins.size()
        bins = bins.view(N, B, H*W)
        bin_centers = 0.5 * (bins[:, 1:, :] + bins[:, :-1, :])
        print(bin_centers.size())
        n, p, l = bin_centers.shape
        input_points = bin_centers.view(n, p, 1, l)  # .shape = n, p, 1
        print(input_points.size())
        # n, c, h, w = target_depth_maps.shape

        print(target_depth_maps.size())
        target_depth_maps = self.partition(target_depth_maps)
        print(target_depth_maps.size())

        # target_points = target_depth_maps.flatten(1, 2)  # n, hwc
        n, p, l = target_depth_maps.size()
        for i in range(l):
            target_points = target_depth_maps[:, :, i]
            mask = target_points > 0  # only valid ground truth points
            target_points = target_points[mask]

            target_lengths = torch.Tensor(target_points.size()).long().to(target_depth_maps.device)
            print(target_lengths)
            target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
       # loss = 0
        return loss
if __name__=='__main__':
    torch.manual_seed(0)
    x = torch.randn((1, 17, 4, 4))
    y = torch.randn((1, 1, 224, 224))

    criterion = BinsChamferLoss_multi()

    loss= criterion(x, y)

    print(loss)
    


