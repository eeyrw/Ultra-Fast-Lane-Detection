import os
import cv2
import tqdm
import numpy as np
import pdb
import json
import argparse
from scripts.convert_tusimple import get_tusimple_list
from data.dataloader import get_test_loader
from evaluation.tusimple.lane import LaneEval
from utils.dist_utils import is_main_process, dist_print, get_rank, get_world_size, dist_tqdm, synchronize
import os
import json
import torch
import scipy
import numpy as np
import platform
from utils.visualize import genSegLabelImage
from data.dataloader import get_gen_pseudo_loader
from model.model import parsingNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='The root of the Tusimple dataset')
    return parser


# def genPseudoSampleList(datasetRoot, outputPath):

#     wantedFrames = ['1', '10']
#     # training set
#     names, _ = get_tusimple_list(args.root,  [
#                                  'label_data_0601.json', 'label_data_0531.json', 'label_data_0313.json'])
#     clipDirList = [os.path.dirname(name) for name in names]
#     wantedClipList = [os.path.join(clipDir, frame+'.jpg')
#                       for clipDir in clipDirList for frame in wantedFrames]
#     wantedClipListGt = [os.path.join(clipDir, frame+'.png')
#                         for clipDir in clipDirList for frame in wantedFrames]

#     with open(outputPath, 'w') as f:
#         for clipPath, labalPath in zip(wantedClipList, wantedClipListGt):
#             f.write(clipPath + ' ' + labalPath+'\n')

def genPseudoGt(net, loader, datasetRoot, segSize, listPath, pseudoGtPath, iter_num):
    print('start generating pseudo ground truth...')
    net.eval()
    progress_bar = dist_tqdm(loader)
    if not os.path.exists(os.path.join(datasetRoot, pseudoGtPath)):
        os.mkdir(os.path.join(datasetRoot, pseudoGtPath))
    with open(os.path.join(datasetRoot, listPath), 'w') as f:
        for b_idx, data_label in enumerate(progress_bar):
            imgs, cls_labels, seg_labels, img_names = data_label
            imgs = imgs.cuda()

            with torch.no_grad():
                out = net(imgs)
                for img, segout, img_path in zip(imgs, out[1], img_names):
                    (fileName, ext) = os.path.splitext(img_path)
                    (path, filenameWithExt) = os.path.split(img_path)
                    label_path = os.path.join(pseudoGtPath, fileName+'.png')
                    if not os.path.exists(os.path.join(datasetRoot, pseudoGtPath, path)):
                        os.makedirs(os.path.join(
                            datasetRoot, pseudoGtPath, path))
                    genSegLabelImage(img, segout, (segSize[1], segSize[0]), os.path.join(
                        datasetRoot, label_path), use_color=False)
                    genSegLabelImage(img, segout, (segSize[1]//2, segSize[0]//2), os.path.join(
                        datasetRoot, pseudoGtPath, fileName+'_%d.webp' % iter_num), use_color=True)
                    f.write(img_path + ' ' + label_path+'\n')

# def genPseudoGt(net, data_root,distributed, batch_size=34):
#     loader = get_gen_pseudo_loader(
#         batch_size, data_root, 'Tusimple', distributed)
#     for i, data in enumerate(dist_tqdm(loader)):
#         imgs, img_paths, label_paths = data
#         imgs = imgs.cuda()
#         with torch.no_grad():
#             out = net(imgs)
#             for img, segout, img_path, label_path in zip(imgs, out[1], img_paths, label_paths):
#                 genSegLabelImage(img, segout, (720, 1280),
#                                  label_path, use_color=False)


if __name__ == "__main__":
    args = get_args().parse_args()
    # reference maskrcnn-benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    datasetName = 'Tusimple'

    if datasetName == 'CULane':
        cls_num_per_lane = 18
        img_w, img_h = 1640, 590
        griding_num = 200
    elif datasetName == 'Tusimple':
        cls_num_per_lane = 56
        img_w, img_h = 1280, 720
        griding_num = 100
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone='res18', cls_dim=(griding_num+1, cls_num_per_lane, 4),
                     use_aux=True).to(device)  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(
        r"..\log\tusimple_b128_ep099.pth", map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])
    # genPseudoSampleList(args.root, os.path.join(args.root, 'train_pseudo_gt.txt'))
    # genPseudoGt(net,args.root,distributed)
