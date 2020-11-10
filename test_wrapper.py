import torch
import os
from model.model import parsingNet
from utils.common import merge_config
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
    assert cfg.backbone in ['18', '34', '50', '101',
                            '152', '50next', '101next', '50wide', '101wide']

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    return eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, testWithAux, distributed, lastMetrics=lastMetrics, proportion=cfg.test_ds_proportion)
