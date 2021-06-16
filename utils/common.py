import datetime
import pathspec
import os
import argparse
from utils.dist_utils import is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import torch
from shutil import copyfile
from defaults import get_cfg_defaults


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--raw_img_size', default=None, type=int, nargs='+')
    parser.add_argument('--train_ds_proportion', default=None, type=float)
    parser.add_argument('--test_ds_proportion', default=None, type=float)
    parser.add_argument('--epoch', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--optimizer', default=None, type=str)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--momentum', default=None, type=float)
    parser.add_argument('--scheduler', default=None, type=str)
    parser.add_argument('--steps', default=None, type=int, nargs='+')
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--warmup', default=None, type=str)
    parser.add_argument('--warmup_iters', default=None, type=int)
    parser.add_argument('--backbone', default=None, type=str)
    parser.add_argument('--griding_num', default=None, type=int)
    parser.add_argument('--use_aux', default=None, type=str2bool)
    parser.add_argument('--use_spp', default=None, type=str2bool)
    parser.add_argument('--sim_loss_w', default=None, type=float)
    parser.add_argument('--shp_loss_w', default=None, type=float)
    parser.add_argument('--note', default=None, type=str)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--finetune', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--test_model', default=None, type=str)
    parser.add_argument('--test_work_dir', default=None, type=str)
    parser.add_argument('--num_lanes', default=None, type=int)
    parser.add_argument('--test_interval', default=None, type=int)
    parser.add_argument('--test_during_train', default=None, type=str2bool)
    return parser


def merge_yacs_config(overrideOpts=''):
    cfg = get_cfg_defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0)
    if overrideOpts != '':
        argsKnown, argsUNKnown = parser.parse_known_args(overrideOpts)
    else:
        argsKnown, argsUNKnown = parser.parse_known_args()
    cfg.merge_from_file(argsKnown.config)
    cfg.merge_from_list(argsUNKnown)
    cfg.freeze()
    return argsKnown, cfg


def merge_config():
    args = get_args().parse_args()
    cfg = Config.fromfile(args.config)

    items = ['dataset', 'data_root', 'raw_img_size', 'train_ds_proportion', 'test_ds_proportion',
             'epoch', 'batch_size', 'optimizer', 'learning_rate',
             'weight_decay', 'momentum', 'scheduler', 'steps', 'gamma', 'warmup', 'warmup_iters',
             'use_aux', 'use_spp', 'griding_num', 'backbone', 'sim_loss_w', 'shp_loss_w', 'note', 'log_path',
             'finetune', 'resume', 'test_model', 'test_work_dir', 'num_lanes', 'test_interval', 'test_during_train']
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))
    return args, cfg


def save_model(net, optimizer, epoch, save_path, distributed):
    if is_main_process():
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict,
                 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
        torch.save(state, model_path)


def save_best_model(net, optimizer, save_path, distributed):
    if is_main_process():
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict,
                 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'best.pth')
        torch.save(state, model_path)


def cp_projects(to_path):
    if is_main_process():
        with open('./.gitignore', 'r') as fp:
            ign = fp.read()
        ign += '\n.git'
        spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, ign.splitlines())
        all_files = {os.path.join(root, name) for root,
                     dirs, files in os.walk('./') for name in files}
        matches = spec.match_files(all_files)
        matches = set(matches)
        to_cp_files = all_files - matches
        # to_cp_files = [f[2:] for f in to_cp_files]
        # pdb.set_trace()
        for f in to_cp_files:
            dirs = os.path.join(to_path, 'code', os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            # os.system('cp %s %s'%(f,os.path.join(to_path,'code',f[2:])))
            copyfile(f, os.path.join(to_path, 'code', f[2:]))


def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.TRAIN.LR, cfg.TRAIN.BATCH_SIZE)
    work_id = now + hyper_param_str + cfg.EXP.NOTE
    work_dir = os.path.join(cfg.EXP.LOG_PATH, work_id)
    return work_dir, work_id


def get_logger(work_dir, cfg):
    logger = DistSummaryWriter(work_dir, flush_secs=60)
    config_txt = os.path.join(work_dir, 'cfg.txt')
    if is_main_process():
        with open(config_txt, 'w') as fp:
            fp.write(str(cfg))

    return logger
