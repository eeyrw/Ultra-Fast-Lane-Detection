from data.constant import tusimple_row_anchor, culane_row_anchor, lane_index_colour
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
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def genPseudoLabelImage(image, segOutput, imageSize, segSize, visualize_path, labelPath):
    # segOutput: [class,h,w]
    segOutput = torch.unsqueeze(torch.unsqueeze(
        torch.argmax(segOutput, dim=0), 0), 0)
    # segOutput: [1,1,h,w]
    plainSegOutput = torch.squeeze(torch.nn.functional.interpolate(
        segOutput.float(), size=segSize)).byte().numpy()
    plainSegOutputWithImageSize = torch.squeeze(torch.nn.functional.interpolate(
        segOutput.float(), size=imageSize)).byte().numpy()        

    image = torch.unsqueeze(image, dim=0)
    image = torch.squeeze(
        torch.nn.functional.interpolate(image.float(), size=imageSize))

    # Step 1. chw to hwc Step 2. RGB to BGR
    img_bgr = normalizeImage(image).transpose((1, 2, 0))[..., ::-1]
    colorMapMat = np.array(
        lane_index_colour, dtype=np.uint8)[..., ::-1]  # RGB to BGR
    segImage = colorMapMat[plainSegOutputWithImageSize]
    res = cv2.addWeighted(img_bgr, 1, segImage, 0.7, 0.4)
    cv2.imwrite(visualize_path, res, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
    cv2.imwrite(labelPath, plainSegOutput)

def genSegLabelImage(image, segOutput, size, path, use_label=True, use_color=False):
    # segOutput: [class,h,w]
    segOutput = torch.unsqueeze(torch.unsqueeze(
        torch.argmax(segOutput, dim=0), 0), 0)
    # segOutput: [1,1,h,w]
    plainSegOutput = torch.squeeze(torch.nn.functional.interpolate(
        segOutput.float(), size=size)).byte().cpu().numpy()

    image = torch.unsqueeze(image, dim=0)
    image = torch.squeeze(
        torch.nn.functional.interpolate(image.float(), size=size))

    if use_color:
        # Step 1. chw to hwc Step 2. RGB to BGR
        img_bgr = normalizeImage(image).transpose((1, 2, 0))[..., ::-1]
        colorMapMat = np.array(
            lane_index_colour, dtype=np.uint8)[..., ::-1]  # RGB to BGR
        segImage = colorMapMat[plainSegOutput]
        res = cv2.addWeighted(img_bgr, 1, segImage, 0.7, 0.4)
        cv2.imwrite(path, res, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
    if use_label:
        cv2.imwrite(path, plainSegOutput)
        # labelImage = Image.fromarray(plainSegOutput)
        # labelImage.save(path)


def logSegLabelImage(logger, tag, step, image, pointOutput, row_anchor, griding_num, cls_num_per_lane, segOutput, size):
    # segOutput: [class,h,w]
    segOutput = torch.unsqueeze(torch.unsqueeze(
        torch.argmax(torch.sigmoid(segOutput), dim=0), 0), 0)
    # segOutput: [1,1,h,w]
    plainSegOutput = torch.squeeze(torch.nn.functional.interpolate(
        segOutput.float(), size=size)).byte().cpu().numpy()

    image = torch.unsqueeze(image, dim=0)
    image = torch.squeeze(
        torch.nn.functional.interpolate(image.float(), size=size))

    # Step 1. chw to hwc Step 2. RGB to BGR
    img_bgr = normalizeImage(image).transpose((1, 2, 0))[..., ::-1]
    colorMapMat = np.array(
        lane_index_colour, dtype=np.uint8)[..., ::-1]  # RGB to BGR
    segImage = colorMapMat[plainSegOutput]
    res = cv2.addWeighted(img_bgr, 1, segImage, 0.7, 0.4)

    img_h = size[0]
    img_w = size[1]
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = pointOutput.data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1)
                    cv2.circle(res, ppp, 4, lane_index_colour[0][::-1], -1)
                    cv2.circle(res, ppp, 3, lane_index_colour[i+1][::-1], -1)

    res = res[..., ::-1].copy()  # BGR to RGB
    logger.add_image(tag, res, step, dataformats='HWC')


def normalizeImage(imageTensor):
    maxVal = torch.max(imageTensor)
    minVal = torch.min(imageTensor)
    imageNormalized = (imageTensor-minVal)/(maxVal-minVal)
    return (imageNormalized*255).byte().cpu().numpy()


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
    genSegLabelImage(torch.randn(3, 720, 1280), torch.randn(
        5, 36, 100), (720, 1280), "a.webp", use_color=True)
