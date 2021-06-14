import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

import math
import torch
import torch.nn.functional as F

from model.spp import SPPLayer
from model.model import parsingNet
from torchstat import stat
import torchvision.models as models
import math
from torchvision import transforms
from PIL import Image
from utils.common import merge_yacs_config
import matplotlib.gridspec as gridspec

class FeatureMapVisualizer(object):
    def __init__(self, model):
        self.model = model
        self._HookModel()
        self.counter = 0
        self.lastLayerName = 'Image input'

    def viz(self, module, input):
        x = input[0][0]
        # 最多显示4张图
        min_num = np.minimum(128, x.size()[0])
        col = int(math.sqrt(min_num))
        col = col if min_num-col*col <= 0 else col+1
        row = col if min_num-col*col <= 0 else col+1
        plt.suptitle(self.lastLayerName,fontsize=5)
        self.lastLayerName = module.__class__
        for i in range(min_num):
            plt.xticks(size = 5)
            plt.yticks(size = 5)
            plt.subplot(row, col, i+1)
            plt.imshow(x[i].cpu().detach().numpy())
        # plt.show()
        plt.savefig('../FeatureMaps/%s_%d.png' % ('sss', self.counter),dpi=200)
        self.counter += 1

    def _HookModel(self):
        for name, m in self.model.named_modules():
            # if not isinstance(m, torch.nn.ModuleList) and \
            #         not isinstance(m, torch.nn.Sequential) and \
            #         type(m) in torch.nn.__dict__.values():
            # 这里只对卷积层的feature map进行显示
            # if isinstance(m, torch.nn.Conv2d):
            print(m)
            m.register_forward_pre_hook(self.viz)


def loadImage(imgPath):
    transFormsForImage = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    rawImageRgb = Image.open(imgPath).convert('RGB')
    return transFormsForImage(rawImageRgb).unsqueeze(0)


def visualizeImageAndLabel(image,output,title):
    maxVal = torch.max(image)
    minVal = torch.min(image)
    imageNormalized = (image-minVal)/(maxVal-minVal)

    maxVal = torch.max(output)
    minVal = torch.min(output)
    outputNormalized = (output.float()-minVal)/(maxVal-minVal)
    outputNormalized = outputNormalized.detach()

    fig2 = plt.figure(constrained_layout=True, figsize=[9, 8], dpi=100)
    spec2 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig2)
    f2_ax1 = fig2.add_subplot(spec2[0, 0])
    f2_ax2 = fig2.add_subplot(spec2[1, 0])
    imageNormalized = imageNormalized.permute(1, 2, 0)  # CHW to HWC
    f2_ax1.set_title("Predict")
    f2_ax1.imshow(imageNormalized, interpolation='bilinear')
    f2_ax1.imshow(outputNormalized[0],alpha = 0.3,cmap=plt.cm.rainbow, vmin=0, vmax=1)
    fig2.savefig('../FeatureMaps/%s.png' % title,dpi=200)
    
def vizFMaps(input,title,counter):
    x = input[0]
    # 最多显示4张图
    min_num = np.minimum(128, x.size()[0])
    col = int(math.sqrt(min_num))
    col = col if min_num-col*col <= 0 else col+1
    row = col if min_num-col*col <= 0 else col+1
    plt.suptitle(title,fontsize=5)
    for i in range(min_num):
        plt.xticks(size = 5)
        plt.yticks(size = 5)
        plt.subplot(row, col, i+1)
        plt.imshow(x[i].cpu().detach().numpy())
    # plt.show()
    plt.savefig('../FeatureMaps/%s_%d.png' % (title, counter),dpi=200)
def genAttnMap(input):
    return torch.softmax(torch.sum(torch.pow(torch.abs(input[0]), 2),dim = 0),dim=0).unsqueeze(dim=0).unsqueeze(dim=0)



if __name__ == '__main__':
    args, cfg = merge_yacs_config(overrideOpts=['configs/culane.yaml',
    'DATASET.TEST_PROPORTION', '0.05',
    'TEST.MODEL', r"D:\LaneDetectExperimentSemiSupervision\log\20210611_200649_lr_1e-01_b_32\best.pth",
    "NETWORK.BACKBONE", "res18",
    "NETWORK.GRIDING_NUM", "50", "NETWORK.USE_ATTN", "False",
    "NETWORK.USE_AUX", "False", "NETWORK.USE_RESA", "False",
    "NETWORK.USE_SFL_ATTN", "False", "DATASET.TRAIN_PROPORTION", "0.04",
    "DATASET.TEST_PROPORTION", "0.05"]
    )
    model= parsingNet(pretrained = False, backbone=cfg.NETWORK.BACKBONE, cls_dim = (cfg.NETWORK.GRIDING_NUM+1, cfg.NETWORK.CLS_NUM_PER_LANE, cfg.DATASET.NUM_LANES),
                    use_aux = False, use_mid_aux = True, fc_mid_chan_num = 128)  # we dont need auxiliary segmentation in testing

    state_dict=torch.load(cfg.TEST.MODEL, map_location = 'cpu')['model']
    compatible_state_dict={}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]]=v
        else:
            compatible_state_dict[k]=v

    model.load_state_dict(compatible_state_dict, strict = False)
    model.eval()
    #vs = FeatureMapVisualizer(model)
    # batchInputs = torch.randn(
    #     1, 3, 128, 256, dtype=torch.float, requires_grad=False)
    img = loadImage(r"E:\CULane\driver_23_30frame\05161256_0556.MP4\00045.jpg")
    b = model(img)
    headMap = torch.nn.functional.interpolate(genAttnMap(b[1]),(288,800),mode="bilinear",align_corners=True).squeeze(dim=0)
    visualizeImageAndLabel(img[0],headMap,'x2')
    # vizFMaps(torch.nn.functional.interpolate(b[0].permute(0, 3,2, 1),(288,800)),"cls_out",0)
    # vizFMaps(genAttnMap(b[1]),"x2",0)
    # vizFMaps(genAttnMap(b[2]),"x3",0)
    # vizFMaps(genAttnMap(b[3]),"x4",0)
    # vizFMaps(genAttnMap(b[4]),"fea",0)

