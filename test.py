import torch
import os
from model.model import parsingNet
from utils.common import merge_yacs_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
from test_wrapper import testNet
import torch

if __name__ == "__main__":
    args, cfg = merge_yacs_config(overrideOpts=['configs/culane.yaml',
    'DATASET.TEST_PROPORTION', '0.05',
    'TEST.MODEL', r'..\log\20210610_211907_lr_1e-01_b_32_attn\best.pth',
    "NETWORK.BACKBONE", "res18",
    "NETWORK.GRIDING_NUM", "50", "NETWORK.USE_ATTN", "True",
    "NETWORK.USE_AUX", "True", "NETWORK.USE_RESA", "False",
    "NETWORK.USE_SFL_ATTN", "True", "DATASET.TRAIN_PROPORTION", "0.04",
    "DATASET.TEST_PROPORTION", "0.05"]
    )

    net= parsingNet(pretrained = False, backbone=cfg.NETWORK.BACKBONE, cls_dim = (cfg.NETWORK.GRIDING_NUM+1, cfg.NETWORK.CLS_NUM_PER_LANE, cfg.DATASET.NUM_LANES),
                    use_aux = False).cuda()  # we dont need auxiliary segmentation in testing

    state_dict=torch.load(cfg.TEST.MODEL, map_location = 'cpu')['model']
    compatible_state_dict={}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]]=v
        else:
            compatible_state_dict[k]=v

    net.load_state_dict(compatible_state_dict, strict = False)

    print(testNet(net, args, cfg, False))
