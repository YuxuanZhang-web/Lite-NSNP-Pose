# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 


class SepConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, bias=True, padding_mode='zeros', depth_multiplier=1):
        super(SepConv2d, self).__init__()

        intermediate_channels = in_channels * depth_multiplier

        self.spatialConv = nn.Conv2d(in_channels, intermediate_channels,kernel_size, stride,
             padding, dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)

        self.pointConv = nn.Conv2d(intermediate_channels, out_channels,
             kernel_size=1, stride=1, padding=0, dilation=1, bias=bias, padding_mode=padding_mode)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.spatialConv(x)
        x = self.relu(x)
        x = self.pointConv(x)

        return x

conv_dict = {
    'CONV2D': nn.Conv2d,
    'SEPARABLE': SepConv2d
}

class _AtrousModule(nn.Module):
    def __init__(self, conv_type, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.conv = conv_dict[conv_type]
        self.atrous_conv = self.conv(inplanes, planes, kernel_size=kernel_size,
                            stride=1, padding=padding, dilation=dilation, bias=False, padding_mode='zeros')

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.relu(x)
        x = self.atrous_conv(x)
        x = self.bn(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class DCASP(nn.Module):
    def __init__(self, conv_type, inplanes, planes, n_classes=16):
        super(DCASP, self).__init__()

        # WASP
        dilations_1 = [1, 2]

        dilations_2 = [1, 4, 8]

        dilations_3 = [1, 6, 12, 18]
        # dilations = [1, 12, 24, 36]

        # convs = conv_dict[conv_type]

        reduction = planes // 8

        BatchNorm = nn.BatchNorm2d
        self.aspp1_1 = _AtrousModule(conv_type, inplanes, planes, 1, padding=0, dilation=dilations_1[0],
                                     BatchNorm=BatchNorm)
        self.aspp2_1 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations_1[1], dilation=dilations_1[1],
                                     BatchNorm=BatchNorm)

        self.aspp1_2 = _AtrousModule(conv_type, inplanes, planes, 1, padding=0, dilation=dilations_2[0],
                                     BatchNorm=BatchNorm)
        self.aspp2_2 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations_2[1], dilation=dilations_2[1],
                                     BatchNorm=BatchNorm)
        self.aspp3_2 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations_2[2], dilation=dilations_2[2],
                                     BatchNorm=BatchNorm)

        self.aspp1_3 = _AtrousModule(conv_type, inplanes, planes, 1, padding=0, dilation=dilations_3[0],
                                     BatchNorm=BatchNorm)
        self.aspp2_3 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations_3[1], dilation=dilations_3[1],
                                     BatchNorm=BatchNorm)
        self.aspp3_3 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations_3[2], dilation=dilations_3[2],
                                     BatchNorm=BatchNorm)
        self.aspp4_3 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations_3[3], dilation=dilations_3[3],
                                     BatchNorm=BatchNorm)

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.ReLU(),
                                             nn.Conv2d(planes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes)
                                             )

        self.conv1_1 = nn.Conv2d(2 * planes, planes, 1, bias=False)
        self.conv1_2 = nn.Conv2d(4 * planes, planes, 1, bias=False)
        self.conv1_3 = nn.Conv2d(5 * planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(reduction)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, reduction, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(48, reduction, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(reduction))
        self.conv4 = nn.Sequential(nn.Conv2d(planes + reduction, planes, kernel_size=3, stride=1, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU())

        self.last_conv = nn.Sequential(nn.Conv2d(planes + reduction, planes, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(),
                                       nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(),
                                       nn.Conv2d(planes, n_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_features):
        residual1 = x
        residual = self.conv3(residual1)

        # x1_1 = self.aspp1_1(x)
        # x2_1 = self.aspp2_1(x1_1)
        # x3_1 = self.global_avg_pool(x2_1)
        # x3_1 = F.interpolate(x3_1, size=x2_1.size()[2:], mode='bilinear', align_corners=True)
        # x = torch.cat((x1_1, x2_1, x3_1), dim=1)
        # x = self.relu(x)
        # x = self.conv1_1(x)
        # x = self.bn1(x)
        #
        #
        #
        x1_2 = self.aspp1_2(x)
        x2_2 = self.aspp2_2(x1_2)
        x3_2 = self.aspp3_2(x2_2)
        x4_2 = self.global_avg_pool(x)
        x4_2 = F.interpolate(x4_2, size=x3_2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1_2, x2_2, x3_2, x4_2), dim=1)
        x = self.relu(x)
        x = self.conv1_2(x)
        x11 = self.bn2(x)




        x1_3 = self.aspp1_3(residual1)
        x2_3 = self.aspp2_3(x1_3)
        x3_3 = self.aspp3_3(x2_3)
        x4_3 = self.aspp4_3(x3_3)
        x5_3 = self.global_avg_pool(residual1)
        x5_3 = F.interpolate(x5_3, size=x4_3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1_3, x2_3, x3_3, x4_3, x5_3), dim=1)
        x = self.relu(x)
        x = self.conv1_3(x)
        x12 = self.bn3(x)

        x = torch.cat((x11, x12), dim=1)
        x = self.relu(x)
        x = self.conv1_1(x)
        x12 = self.bn1(x)

        x = torch.cat((x12, residual),dim=1)
        x = self.conv4(x)
        # x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        low_level_features = self.relu(low_level_features)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn4(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x
