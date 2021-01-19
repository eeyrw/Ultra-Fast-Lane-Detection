import torch
import os
import datetime
import numpy as np

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
import pickle
import time


def genEmbedding(net, loader, name):
    print('start generating pseudo ground truth...')
    net.eval()
    progress_bar = dist_tqdm(loader)
    embeddingDict = {'name': name, 'embedding': {}}
    for b_idx, data_label in enumerate(progress_bar):
        imgs, _, img_names = data_label
        imgs = imgs.cuda()

        with torch.no_grad():
            out = net(imgs)
            for mid_aux_out, img_path in zip(out[1], img_names):
                embeddingDict['embedding'][img_path] = mid_aux_out.cpu().numpy()
    with open('DatasetEmbedding_%s.pkl' % name, 'wb') as f:
        pickle.dump(embeddingDict, f)


def genEmbeddingTusimple():
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_yacs_config(
        overrideOpts=[r"D:\LaneDetectExperimentSemiSupervision\configs\culane.yaml"])

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    dist_print(cfg)
    assert cfg.NETWORK.BACKBONE in validBackbones

    cls_num_per_lane = cfg.NETWORK.CLS_NUM_PER_LANE

    net = parsingNet(pretrained=True, backbone=cfg.NETWORK.BACKBONE, cls_dim=(
        cfg.NETWORK.GRIDING_NUM+1, cls_num_per_lane, cfg.DATASET.NUM_LANES), use_aux=False, use_spp=False, use_mid_aux=True).cuda()

    weight_file_path = r"D:\LaneDetectExperimentSemiSupervision\log\culane_ep049.pth"
    state_all = torch.load(weight_file_path)['model']
    state_clip = {}  # only use backbone parameters
    for k, v in state_all.items():
        if 'model' in k:
            state_clip[k] = v
    net.load_state_dict(state_clip, strict=False)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])

    loader, _ = get_train_loader(
        35, 'E:/Tusimple',
        100, 'Tusimple-no-aug',
            False, distributed,
        4, 1,
        load_name=True, pin_memory=True,
        num_workers=4
    )
    genEmbedding(net, loader, 'Tusimple')


def genEmbeddingCULane():
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_yacs_config(
        overrideOpts=[r"D:\LaneDetectExperimentSemiSupervision\configs\tusimple.yaml"])

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    dist_print(cfg)
    assert cfg.NETWORK.BACKBONE in validBackbones

    cls_num_per_lane = cfg.NETWORK.CLS_NUM_PER_LANE

    net = parsingNet(pretrained=True, backbone=cfg.NETWORK.BACKBONE, cls_dim=(
        cfg.NETWORK.GRIDING_NUM+1, cls_num_per_lane, cfg.DATASET.NUM_LANES), use_aux=False, use_spp=False, use_mid_aux=True).cuda()

    weight_file_path = r"D:\LaneDetectExperimentSemiSupervision\log\tusimple_b128_ep099.pth"
    state_all = torch.load(weight_file_path)['model']
    state_clip = {}  # only use backbone parameters
    for k, v in state_all.items():
        if 'model' in k:
            state_clip[k] = v
    net.load_state_dict(state_clip, strict=False)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])

    loader, _ = get_train_loader(
        35, 'E:/CULane',
        100, 'CULane-no-aug',
            False, distributed,
        4, proportion=1,
        load_name=True, pin_memory=True,
        num_workers=4
    )
    genEmbedding(net, loader, 'CULane')


if __name__ == "__main__":
    genEmbeddingCULane()
    genEmbeddingTusimple()
