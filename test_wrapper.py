import torch
import os
from model.model import parsingNet
from model.model import validBackbones
from utils.common import merge_yacs_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch


def testNet(net, args, cfg, testWithAux, lastMetrics=None):
    torch.backends.cudnn.benchmark = True

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    dist_print('start testing...')
    assert cfg.NETWORK.BACKBONE in validBackbones

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])

    if not os.path.exists(cfg.TEST.WORK_DIR):
        os.mkdir(cfg.TEST.WORK_DIR)

    return eval_lane(net, cfg.DATASET.NAME, cfg.DATASET.ROOT, cfg.TEST.WORK_DIR, cfg.NETWORK.GRIDING_NUM, testWithAux, distributed, lastMetrics=lastMetrics, batch_size=cfg.TEST.BATCH_SIZE, proportion=cfg.DATASET.TEST_PROPORTION)
