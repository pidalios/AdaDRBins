import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# basic blocks ##########################################################################################################################################
def weights_init(m):
    # initialize kernel weights with gaussian distributions
    if isinstance(m, nn.Conv2d):
       n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
       m.weight.data.normal_(0, math.sqrt(2. / n))
       if m.bias is not None:
           m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
       m.weight.data.fill_(1)
       m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
       m.weight.data.normal_(0, 0.01)
       m.bias.data.zero_()

def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)
    
def convbn_relu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels), 
        nn.ReLU(inplace=True)
	)
def convbn_dw_relu(in_channels, kernel_size):
    padding = (kernel_size-1)//2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU6(inplace=True),
        )

def convbn_pw_relu(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU6(inplace=True),
        )

def convbn_3x3dw_relu(in_channels, kernel_size):
    padding = (kernel_size-1)//2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )
def convbn_1x1_relu(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def depthwise_separable_conv(inp, oup, kernel_size):
    return nn.Sequential(
                convbn_3x3dw_relu(inp, kernel_size), 
                convbn_1x1_relu(inp, oup)
                )
############################################################################################################################################################
# For mobilenet v3 #########################################################################################################################################
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

############################################################################################################################################################
# for CSPN #################################################################################################################################################
def kernel_trans(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size-1)/2))
    # print(kernel.size())
    return kernel



class CSPNGenerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        #normalization in standard CSPN
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output =  torch.cat((half1, guide_mid, half2), dim=1)
        return output

class CSPN(nn.Module):
    def __init__(self, kernel_size, dilation=1, stride=1):
        super(CSPN, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size-1)//2
        self.stride = stride

    def forward(self, kernel, input, input0): #with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size*self.kernel_size-1)/2)
        input_im2col[:, mid_index:mid_index+1, :] = input0

        #print(input_im2col.size(), kernel.size())
        cspn_output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel)).view(bs, 1, h, w)
        # output = iter_weight * cspn_output.view(bs, 1, h, w) + input
        return cspn_output

class CSPN_original(nn.Module):
    def __init__(self, kernel_size, dilation=1, stride=1):
        super(CSPN_original, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size-1)//2
        self.stride = stride

    def forward(self, kernel, iter_weight, input, input0): #with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size*self.kernel_size-1)/2)
        input_im2col[:, mid_index:mid_index+1, :] = input0

        #print(input_im2col.size(), kernel.size())
        cspn_output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel)).view(bs, 1, h, w)
        output = iter_weight * cspn_output + input
        return cspn_output

############################################################################################################################################################
# Up blocks ################################################################################################################################################
class UpCSPN(nn.Module):
    def __init__(self, inp, oup, k=3):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.kernel_size = k

        self.relu = nn.ReLU(inplace=True)

        self.kernel_generation = CSPNGenerate(inp, k)
        encoder = torch.zeros(k * k, k * k, k, k)
        self.encoder = nn.Parameter(encoder, requires_grad=True)
        self.CSPN = CSPN(kernel_size=k)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = convbn_relu(inp, inp, kernel_size=5,stride=1, padding=2)
        self.conv2 = convbn_relu(inp, 1, kernel_size=1,stride=1, padding=0)
        self.conv3 = convbn_relu(inp, oup - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, guide, depth):
        x = guide + depth + x
        x = self.conv1(x)
        cspn_x = x

        cspn_depth = self.conv2(x)

        guide_kernel = self.kernel_generation(cspn_x)
        guide_kernel = kernel_trans(guide_kernel, self.encoder)
        guided_result = self.CSPN(guide_kernel, cspn_depth, cspn_depth)
        x = self.conv3(x)

        x = torch.cat([guided_result, x], dim=1)
        x = self.upsample(x)
        return x

class UpCSPN_n(nn.Module):
    def __init__(self, inp, oup, k=3, n=2):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.kernel_size = k
        self.n = n

        self.relu = nn.ReLU(inplace=True)

        self.kernel_generation = CSPNGenerate(inp, k)
        encoder = torch.zeros(k * k, k * k, k, k)
        self.encoder = nn.Parameter(encoder, requires_grad=True)
        self.CSPN = CSPN(kernel_size=k)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = convbn_relu(inp, inp, kernel_size=5,stride=1, padding=2)
        self.conv2 = convbn_relu(inp, 1, kernel_size=1,stride=1, padding=0)
        self.conv3 = convbn_relu(inp, oup - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, guide, depth):
        x = guide + depth + x
        x = self.upsample(x)
        x = self.conv1(x)
        cspn_x = x

        cspn_depth = self.conv2(x)
        init_depth = cspn_depth

        guide_kernel = self.kernel_generation(cspn_x)
        guide_kernel = kernel_trans(guide_kernel, self.encoder)
        for i in range(self.n):
            cspn_depth = self.CSPN(guide_kernel, cspn_depth, init_depth)
        x = self.conv3(x)

        x = torch.cat([cspn_depth, x], dim=1)
        return x, cspn_depth

class UpProj(nn.Module):
    def __init__(self, inp, oup, k1=5, k2=3):
        super().__init__()
        pad1 = (k1 - 1)//2
        pad2 = (k2 - 1)//2
        self.branch1 = nn.Sequential(convbn(inp, inp, k1, padding=pad1), 
                                     nn.ReLU(inplace=True), 
                                     convbn(inp, oup, k2, padding=pad2))
        self.branch2 = convbn(inp, oup, k1, padding=pad1)
        self.relu = nn.ReLU(inplace=True)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, r, d):
        x = x + r + d
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        res_x = self.branch2(x)
        x = self.branch1(x) + res_x
        x = self.relu(x)
        return x

class simpleUp(nn.Module):
    def __init__(self, inp, oup, k=5):
        super().__init__()
        pad = (k-1)//2
        self.conv = depthwise_separable_conv(inp, oup, k)

    def forward(self, r, d, x=None):
        if x != None:
            x = x + r + d
        else:
            x = r + d
        x = self.conv(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

class InvertedResidualUp(nn.Module):
    def __init__(self, inp, hidden_dim, oup, k, stride, use_se, use_hs):
        super().__init__()
        pad = (k-1)//2
        self.conv = InvertedResidual(inp, hidden_dim, oup, k, stride, use_se, use_hs)

    def forward(self, x, r, d):
        x = x + r + d
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x




if __name__ == '__main__':
    feature = torch.randn(16, 256, 28, 28)
    depth = torch.randn(16, 256, 28, 28)


