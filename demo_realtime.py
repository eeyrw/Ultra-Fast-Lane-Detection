import torch
import os
import cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special
import tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def resizeAndCropToTargetSize(img, width, height):
    rawW, rawH = img.size
    rawAspectRatio = rawW/rawH
    wantedAspectRatio = width/height
    if rawAspectRatio > wantedAspectRatio:
        scaleFactor = height/rawH
        widthBeforeCrop = int(rawW*scaleFactor)
        return img.resize((widthBeforeCrop, height), Image.BILINEAR). \
            crop(((widthBeforeCrop-width)//2, 0,
                  (widthBeforeCrop-width)//2+width, height))
    else:
        scaleFactor = width/rawW
        heightBeforeCrop = int(rawH*scaleFactor)
        return img.resize((width, heightBeforeCrop), Image.BILINEAR). \
            crop((0, (heightBeforeCrop-height)//2, width,
                  (heightBeforeCrop-height)//2+height))


if __name__ == "__main__":

    # reference maskrcnn-benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101',
                            '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num+1, cls_num_per_lane, 4),
                     use_aux=False).to(device)  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        # transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    videFilePath = 'rain.mp4'
    writeToVideo = False

    cap = cv2.VideoCapture(videFilePath)
    if writeToVideo:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(videFilePath[:-3]+'avi')
        vout = cv2.VideoWriter(
            videFilePath[:-3]+'avi', fourcc, 30.0, (1640, 590))
    while (cap.isOpened()):
        for i in range(1):
            ret, frame = cap.read()
        if frame is None:
            break

        imageRgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imageRgb = resizeAndCropToTargetSize(imageRgb, 1640, 590)
        imageBgrCV = cv2.cvtColor(np.asarray(imageRgb), cv2.COLOR_RGB2BGR)
        imageRgb = resizeAndCropToTargetSize(imageRgb, 800, 288)
        imageTensor = img_transforms(imageRgb)
        imgs = torch.unsqueeze(imageTensor, 0).to(device)
        with torch.no_grad():
            out = net(imgs)

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        lane_index_colour =[(0, 0, 0),(192, 57, 43),(41, 128, 185),(22, 160, 133),(243, 156, 18)]

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                               int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1)
                        cv2.circle(imageBgrCV, ppp, 5, lane_index_colour[i][::-1], -1)
        if writeToVideo:
            vout.write(imageBgrCV)
        cv2.imshow('L', imageBgrCV)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()
    if writeToVideo:
        vout.release()
