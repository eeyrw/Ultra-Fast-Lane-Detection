import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
from test_wrapper import testNet
import torch

if __name__ == "__main__":
    args, cfg = merge_config()

    if cfg.DATASET.NAME == 'CULane':
        cls_num_per_lane = 18
    elif cfg.DATASET.NAME == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.NETWORK.BACKBONE,cls_dim = (cfg.NETWORK.GRIDING_NUM+1,cls_num_per_lane, cfg.DATASET.NUM_LANES),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.TEST.MODEL, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = False)

    print(testNet(net,args,cfg,False))