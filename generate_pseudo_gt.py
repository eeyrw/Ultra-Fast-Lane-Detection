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
from utils.visualize import genPseudoLabelImage
from data.dataloader import get_gen_pseudo_loader
from model.model import parsingNet
from multiprocessing import Pool
from shutil import copyfile


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


def genPseudoGt(net, loader, datasetRoot, segSize, listPath, pseudoGtPath, iter_num, multiproc_num=8, use_multiproc=False):
    print('start generating pseudo ground truth...')
    net.eval()
    progress_bar = dist_tqdm(loader)
    if not os.path.exists(os.path.join(datasetRoot, pseudoGtPath)):
        os.mkdir(os.path.join(datasetRoot, pseudoGtPath))
    listTruePath = os.path.join(datasetRoot, listPath)
    listKeepPath = os.path.join(datasetRoot, pseudoGtPath, listPath)
    with open(listTruePath, 'w') as f:
        if use_multiproc:
            pool = Pool(multiproc_num)  # 创建拥有3个进程数量的进程池
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
                    visualPath = os.path.join(
                        datasetRoot, pseudoGtPath, fileName+'_%d.webp' % iter_num)
                    labelPath = os.path.join(datasetRoot, label_path)
                    imgSize = (segSize[1]//2, segSize[0]//2)
                    segOutImgSize = (segSize[1], segSize[0])
                    if use_multiproc:
                        pool.apply(genPseudoLabelImage,
                                         (img.cpu(), segout.cpu(), imgSize, segOutImgSize, visualPath, labelPath))
                    else:
                        genPseudoLabelImage(img.cpu(), segout.cpu(
                        ), imgSize, segOutImgSize, visualPath, labelPath)
                    # genSegLabelImage(img, segout, (segSize[1], segSize[0]), os.path.join(
                    #     datasetRoot, label_path), use_color=False)
                    # genSegLabelImage(img, segout, (segSize[1]//2, segSize[0]//2), os.path.join(
                    #     datasetRoot, pseudoGtPath, fileName+'_%d.webp' % iter_num), use_color=True)
                    f.write(img_path + ' ' + label_path+'\n')
        copyfile(listTruePath, listKeepPath)
        if use_multiproc:
            pool.close()  # 关闭进程池，不再接受新的进程
            pool.join()  # 主进程阻塞等待子进程的退出

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

    from model.model import parsingNet
    from model.model import validBackbones
    from data.dataloader import get_train_loader

    from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
    from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
    from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

    from utils.common import merge_yacs_config, save_model, save_best_model, cp_projects
    from utils.common import get_work_dir, get_logger

    from test_wrapper import testNet
    from utils.visualize import genSegLabelImage, logSegLabelImage
    from data.constant import tusimple_row_anchor, culane_row_anchor
    from generate_pseudo_gt import genPseudoGt
    # reference maskrcnn-benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"

    datasetName = 'Tusimple'
    distributed = False
    args, cfg = merge_yacs_config(overrideOpts=['../configs/tusimple.yaml'])

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

    train_loaders_no_aug, _ = get_train_loader(
        80, cfg.DATASET.ROOT,
        cfg.NETWORK.GRIDING_NUM, cfg.DATASET.NAME+'-no-aug',
        cfg.NETWORK.USE_AUX, distributed,
        cfg.DATASET.NUM_LANES, 0.8, split=True,
        split_proportion=0.5,
        load_name=True, pin_memory=cfg.DATASET.PIN_MEMORY,
        num_workers=cfg.DATASET.NUM_WORKERS
    )
    pseudo_gen_loader = train_loaders_no_aug[1]
    genPseudoGt(net, pseudo_gen_loader, cfg.DATASET.ROOT, cfg.DATASET.RAW_IMG_SIZE,
                "train_pseudo_gt.txt", "pseudo_clips_gt", 1, use_multiproc=False)
