import torch
import os
from model.model import parsingNet
from model.model import validBackbones
from utils.common import merge_yacs_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch
import shutil


def testNet(net, args, cfg, testWithAux, lastMetrics=None, work_id=None):
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

    if work_id is not None:
        testTmpDir = os.path.join(cfg.TEST.WORK_DIR, work_id)
    else:
        testTmpDir = cfg.TEST.WORK_DIR
    if not os.path.exists(testTmpDir):
        os.mkdir(testTmpDir)

    return eval_lane(net, cfg.DATASET.NAME, cfg.DATASET.ROOT, testTmpDir, cfg.NETWORK.GRIDING_NUM, testWithAux, cfg.NETWORK.USE_MID_AUX, distributed, lastMetrics=lastMetrics, batch_size=cfg.TEST.BATCH_SIZE, proportion=cfg.DATASET.TEST_PROPORTION)


def testRemoveTemps(cfg, work_id):
    testTmpDir = os.path.join(cfg.TEST.WORK_DIR, work_id)
    if os.path.exists(testTmpDir):
        shutil.rmtree(testTmpDir)
