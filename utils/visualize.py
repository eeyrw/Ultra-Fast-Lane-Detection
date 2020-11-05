from torchvision import transforms
from PIL import Image
import scipy.io
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import shutil
import datetime
import argparse
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib


def genSegLabelImage(segOutput,size,path):
    # segOutput: [class,h,w]
    segOutput = torch.unsqueeze(torch.unsqueeze(torch.argmax(torch.sigmoid(segOutput),dim=0), 0), 0)
    # segOutput: [1,1,h,w]
    plainSegOutput = torch.squeeze(torch.nn.functional.interpolate(segOutput.float(),size=size)).byte()
    print(plainSegOutput)
    labelImage = Image.fromarray(plainSegOutput.cpu().numpy())
    labelImage.save(path)
    

def visualizeImageAndLabel(self, writer, tag, step, image, label, output):
    maxVal = torch.max(image)
    minVal = torch.min(image)
    imageNormalized = (image-minVal)/(maxVal-minVal)
    maxVal = torch.max(label)
    minVal = torch.min(label)
    labelNormalized = (label.float()-minVal)/(maxVal-minVal)
    maxVal = torch.max(output)
    minVal = torch.min(output)
    outputNormalized = (output.float()-minVal)/(maxVal-minVal)
    outputNormalized = outputNormalized.detach()
    # writer.add_image('DsInspect/In',imageNormalized, 0, dataformats='CHW')
    # writer.add_images('DsInspect/Label_Out', torch.stack((labelNormalized,outputNormalized)), 0, dataformats='NCHW')
    # writer.add_image('DsInspect/Out', outputNormalized.unsqueeze(0), 0, dataformats='CHW')
    fig2 = plt.figure(constrained_layout=True, figsize=[9, 8], dpi=100)
    spec2 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig2)
    f2_ax1 = fig2.add_subplot(spec2[0, 0])
    f2_ax2 = fig2.add_subplot(spec2[1, 0])
    imageNormalized = imageNormalized.permute(1, 2, 0)  # CHW to HWC
    f2_ax1.set_title("Predict")
    f2_ax1.imshow(imageNormalized, interpolation='bilinear')
    f2_ax1.imshow(outputNormalized[0], alpha=outputNormalized[0]
                  * 0.7, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    f2_ax2.set_title("Ground Truth")
    f2_ax2.imshow(imageNormalized, interpolation='bilinear')
    f2_ax2.imshow(labelNormalized[0], alpha=labelNormalized[0]
                  * 0.7, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    writer.add_figure(tag, fig2,
                      global_step=step, close=True, walltime=None)



if __name__ == "__main__":
    genSegLabelImage(torch.rand(5,6,7),(110,98),"a.png")
