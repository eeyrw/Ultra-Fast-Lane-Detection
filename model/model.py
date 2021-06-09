import torch
from model.backbone import resnet
from model.spp import SPPLayer
from model.fast_scnn import FastSCNN
from model.efficientnetv2 import effnetv2_s
from model.selfAttention import Self_Attn
from model.resa import RESA
from model.shuffle_attention import sa_layer
import numpy as np

validBackbones = ['effnetv2', 'fast_scnn', 'res18', 'res34', 'res50', 'res101',
                  'res152', '50next', '101next', '50wide', '101wide']


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='res50', cls_dim=(37, 10, 4),
                 use_aux=False, use_spp=False, use_attn=False, use_resa=False, use_sfl_attn=False, use_mid_aux=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        # (num_gridding, num_cls_per_lane, num_of_lanes)
        self.cls_dim = cls_dim
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.use_spp = use_spp
        self.use_attn = use_attn
        self.use_resa = use_resa
        self.use_sfl_attn = use_sfl_attn
        self.use_mid_aux = use_mid_aux
        self.total_dim = np.prod(cls_dim)
        self.backbone = backbone

        if self.backbone == 'fast_scnn':
            self.model = FastSCNN(
                5, segOut=use_aux, segOutSize=(36, 100), midFeature=True)

            self.pool = torch.nn.Sequential(
                torch.nn.AvgPool2d(4),
                torch.nn.Conv2d(128, 8, 1),
            )
            self.interPoolChnNum = 8*100*36//16

            self.cls = torch.nn.Sequential(
                torch.nn.Linear(self.interPoolChnNum, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, self.total_dim),
            )
            # 1/32,2048 channel
            # 288,800 -> 9,40,2048
            # (w+1) * sample_rows * 4
            # 37 * 10 * 4
            initialize_weights(self.model)
            initialize_weights(self.cls)

        elif self.backbone == 'effnetv2':
            self.model = effnetv2_s(
                segOutChanNum=5, segOut=use_aux, segOutSize=(36, 100), midFeature=True)

            self.pool = torch.nn.Sequential(
                torch.nn.AvgPool2d(4)
            )
            self.interPoolChnNum = 272*(25//4)*(9//4)

            self.cls = torch.nn.Sequential(
                # torch.nn.Linear(self.interPoolChnNum, 512),
                # torch.nn.ReLU(),
                torch.nn.Linear(self.interPoolChnNum, self.total_dim),
            )
            # 1/32,2048 channel
            # 288,800 -> 9,40,2048
            # (w+1) * sample_rows * 4
            # 37 * 10 * 4
            initialize_weights(self.model)
            initialize_weights(self.cls)
        else:
            # input : nchw,
            # output: (w+1) * sample_rows * 4
            self.model = resnet(backbone, pretrained=pretrained)

            if self.use_aux:
                self.aux_header2 = torch.nn.Sequential(
                    conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in [
                        'res34', 'res18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(128, 128, 3, padding=1),
                    conv_bn_relu(128, 128, 3, padding=1),
                    conv_bn_relu(128, 128, 3, padding=1),
                )
                self.aux_header3 = torch.nn.Sequential(
                    conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in [
                        'res34', 'res18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(128, 128, 3, padding=1),
                    conv_bn_relu(128, 128, 3, padding=1),
                )
                self.aux_header4 = torch.nn.Sequential(
                    conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in [
                        'res34', 'res18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(128, 128, 3, padding=1),
                )
                self.aux_combine = torch.nn.Sequential(
                    conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                    conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                    conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                    conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                    torch.nn.Conv2d(128, cls_dim[-1] + 1, 1)
                    # output : n, num_of_lanes+1, h, w
                )
                initialize_weights(
                    self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

            self.interPoolChnNum = 1800

            if self.use_attn or self.use_sfl_attn:
                self.cls = torch.nn.Sequential(
                    torch.nn.Linear(self.interPoolChnNum, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, self.total_dim),
                )
            else:
                self.cls = torch.nn.Sequential(
                    torch.nn.Linear(self.interPoolChnNum, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, self.total_dim),
                )

            self.pool = torch.nn.Conv2d(512, 8, 1) if backbone in [
                'res34', 'res18'] else torch.nn.Conv2d(2048, 8, 1)
            # 1/32,2048 channel
            # 288,800 -> 9,40,2048
            # (w+1) * sample_rows * 4
            # 37 * 10 * 4
            initialize_weights(self.cls)

            if self.use_attn:
                self.selfAttn = Self_Attn(512)
                initialize_weights(self.selfAttn)

            if self.use_sfl_attn:
                self.sfl_attn = sa_layer(512)
                initialize_weights(self.sfl_attn)

            if self.use_resa:
                self.resa = RESA(512, 9, 25)
                # initialize_weights(self.resa)

            self.avgPool = torch.nn.Sequential(torch.nn.Conv2d(512, self.total_dim, 1),
                                               torch.nn.ReLU(),
                                               torch.nn.AdaptiveAvgPool2d((1, 1)))
            initialize_weights(self.avgPool)

    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        if self.backbone == 'fast_scnn':
            if self.use_aux:
                aux_seg, fea = self.model(x)
            else:
                fea = self.model(x)[0]

            fea = self.pool(fea)
            mid_fea = fea.view(-1, self.interPoolChnNum)

            group_cls = self.cls(mid_fea)
            group_cls = group_cls.view(-1, *self.cls_dim)

            if self.use_aux:
                return group_cls, aux_seg
            if self.use_mid_aux:
                return group_cls, mid_fea

            return group_cls
        elif self.backbone == 'effnetv2':
            if self.use_aux:
                aux_seg, fea = self.model(x)
            else:
                fea = self.model(x)[0]

            fea = self.pool(fea)
            mid_fea = fea.view(-1, self.interPoolChnNum)

            group_cls = self.cls(mid_fea)
            group_cls = group_cls.view(-1, *self.cls_dim)

            if self.use_aux:
                return group_cls, aux_seg
            if self.use_mid_aux:
                return group_cls, mid_fea

            return group_cls
        else:
            x2, x3, fea = self.model(x)

            if self.use_aux:
                x2 = self.aux_header2(x2)
                x3 = self.aux_header3(x3)
                x3 = torch.nn.functional.interpolate(
                    x3, scale_factor=2, mode='bilinear', align_corners=True)
                x4 = self.aux_header4(fea)
                x4 = torch.nn.functional.interpolate(
                    x4, scale_factor=4, mode='bilinear', align_corners=True)
                aux_seg = torch.cat([x2, x3, x4], dim=1)
                aux_seg = self.aux_combine(aux_seg)
            else:
                aux_seg = None

            if self.use_attn:
                fea = self.selfAttn(fea)[0]

            if self.use_resa:
                fea = self.resa(fea)

            if self.use_sfl_attn:
                fea = self.sfl_attn(fea)

            fea = self.pool(fea)
            mid_fea = fea.view(-1, self.interPoolChnNum)

            group_cls = self.cls(mid_fea)

            # group_cls = self.avgPool(fea)
            group_cls = group_cls.view(-1, *self.cls_dim)

            if self.use_aux:
                return group_cls, aux_seg
            if self.use_mid_aux:
                return group_cls, mid_fea

            return group_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
