import torch
import os
import datetime
import numpy as np
import math

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

import time


def inference(net, data_label, use_aux, load_name):
    if use_aux and not load_name:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.cuda(), seg_label.cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}
    elif load_name and not use_aux:
        img, cls_label, img_name = data_label
        img, cls_label = img.cuda(), cls_label.cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'img_name': img_name}
    elif use_aux and load_name:
        img, cls_label, seg_label, img_name = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.cuda(), seg_label.cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label, 'img_name': img_name}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar(
                'loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, net_teacher, data_loader, loss_dict, optimizer, scheduler, logger,
          epoch, metric_dict, use_aux, cfg, cls_num_per_lane):
    net.train()
    net_teacher.eval()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)

        global_batch_step = (epoch * len(data_loader) + b_idx)
        global_sample_iter = global_batch_step * cfg.TRAIN.BATCH_SIZE

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux, load_name=True)
        results_teacher = inference(net_teacher, data_label, use_aux, load_name=True)
        if global_batch_step % 200 == 0:
            if cfg.DATASET.NAME == 'CULane':
                cls_num_per_lane = 18
                img_w, img_h = 1640, 590
                row_anchor = culane_row_anchor
            elif cfg.DATASET.NAME == 'Tusimple':
                cls_num_per_lane = 56
                img_w, img_h = 1280, 720
                row_anchor = tusimple_row_anchor

            if 'seg_out' not in results:
                results['seg_out'] = [None]

            logSegLabelImage(logger,
                             'predImg',
                             global_sample_iter,
                             data_label[0][0],
                             results['cls_out'][0],
                             row_anchor,
                             cfg.NETWORK.GRIDING_NUM,
                             cls_num_per_lane,
                             results['seg_out'][0],
                             (img_h//2, img_w//2))

        loss = calc_loss(loss_dict, results, logger, global_sample_iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_sample_iter)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)

        logger.add_scalar(
            'meta/epoch', epoch, global_sample_iter)

        if global_batch_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name,
                                  me_op.get(), global_step=global_sample_iter)
        logger.add_scalar(
            'meta/lr', optimizer.param_groups[0]['lr'], global_step=global_sample_iter)

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(
                metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss='%.3f' % float(loss),
                                     data_time='%.3f' % float(
                                         t_data_1 - t_data_0),
                                     net_time='%.3f' % float(
                                         t_net_1 - t_net_0),
                                     **kwargs)
        if math.isnan(float(loss)):
            print('The loss turns into NaN. Terminate training.')
            break

        t_data_0 = time.time()


def recoveryState(net, optimizer, cfg):
    if cfg.EXP.FINETUNE is not None:
        dist_print('finetune from ', cfg.EXP.FINETUNE)
        state_all = torch.load(cfg.EXP.FINETUNE)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.EXP.RESUME is not None:
        dist_print('==> Resume model from ' + cfg.EXP.RESUME)
        resume_dict = torch.load(cfg.EXP.RESUME, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        # resume_epoch = int(os.path.split(cfg.EXP.RESUME)[1][2:5]) + 1
        resume_epoch = 0
    else:
        resume_epoch = 0

    return resume_epoch


def getTrainLoader(args, cfg):
    train_loader, cls_num_per_lane = get_train_loader(
        cfg.TRAIN.BATCH_SIZE, cfg.DATASET.ROOT,
        cfg.NETWORK.GRIDING_NUM, cfg.DATASET.NAME,
        cfg.NETWORK.USE_AUX, distributed,
        cfg.DATASET.NUM_LANES, cfg.DATASET.TRAIN_PROPORTION,
        load_name=True, pin_memory=cfg.DATASET.PIN_MEMORY,
        num_workers=cfg.DATASET.NUM_WORKERS
    )
    return train_loader


def getOptimizerAndSchedulerAndResumeEpoch(paramSet, net, loader, cfg):
    optimizer = get_optimizer(net, cfg, paramSet=paramSet)
    resume_epoch = recoveryState(net, optimizer, cfg)
    scheduler = get_scheduler(
        optimizer, cfg, len(loader) * cfg.TRAIN.BATCH_SIZE, paramSet=paramSet)
    return optimizer, scheduler, resume_epoch


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)

    args_student, cfg_student = merge_yacs_config()
    args_teacher, cfg_teacher = merge_yacs_config(
        overrideOpts=[r'..\log\20210613_210527_lr_1e-01_b_60cl_use_res50\cfg.yaml',
                      'EXP.RESUME', r'..\log\20210613_210527_lr_1e-01_b_60cl_use_res50\best.pth']
    )

    work_dir = get_work_dir(cfg_student)
    currentDateTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args_student.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime(
        '[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg_student)
    assert cfg_student.NETWORK.BACKBONE in validBackbones

    cls_num_per_lane = cfg_student.NETWORK.CLS_NUM_PER_LANE

    net_teacher = parsingNet(pretrained=False, backbone=cfg_teacher.NETWORK.BACKBONE, cls_dim=(
        cfg_teacher.NETWORK.GRIDING_NUM+1, cls_num_per_lane, cfg_teacher.DATASET.NUM_LANES),
        use_aux=False, use_spp=cfg_teacher.NETWORK.USE_SPP,use_mid_aux = True,
        use_attn=cfg_teacher.NETWORK.USE_ATTN, use_resa=cfg_teacher.NETWORK.USE_RESA,
        use_sfl_attn=cfg_teacher.NETWORK.USE_SFL_ATTN).cuda()

    net_student = parsingNet(pretrained=True, backbone=cfg_student.NETWORK.BACKBONE, cls_dim=(
        cfg_student.NETWORK.GRIDING_NUM+1, cls_num_per_lane, cfg_student.DATASET.NUM_LANES),
        use_aux=False, use_spp=cfg_student.NETWORK.USE_SPP,use_mid_aux = True,
        use_attn=cfg_student.NETWORK.USE_ATTN, use_resa=cfg_student.NETWORK.USE_RESA,
        use_sfl_attn=cfg_student.NETWORK.USE_SFL_ATTN).cuda()

    if distributed:
        net_teacher = torch.nn.parallel.DistributedDataParallel(
            net_teacher, device_ids=[args_student.local_rank])
        net_student = torch.nn.parallel.DistributedDataParallel(
            net_student, device_ids=[args_student.local_rank])

    train_loader = getTrainLoader(args_student, cfg_student)
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg_student)
    loss_dict = get_loss_dict(cfg_student)
    logger = get_logger(work_dir, cfg_student)
    cp_projects(work_dir)

    bestMetrics = None
    logger.add_text('configuration', str(cfg_student))

    optmzr, scdulr, resume_epoch = getOptimizerAndSchedulerAndResumeEpoch(
        'TRAIN', net_student, train_loader, cfg_student)

    getOptimizerAndSchedulerAndResumeEpoch(
        'TRAIN', net_teacher, train_loader, cfg_teacher)

    for epoch in range(resume_epoch, cfg_student.TRAIN.EPOCH):

        train(net_student, net_teacher, train_loader, loss_dict, optmzr, scdulr,
              logger, epoch, metric_dict, cfg_student.NETWORK.USE_AUX, cfg_student, cls_num_per_lane)
        if cfg_student.TEST.DURING_TRAIN and (epoch % cfg_student.TEST.INTERVAL == 0):
            metricsDict, isBetter = testNet(
                net_student, args_student, cfg_student, True, lastMetrics=bestMetrics)
            sampleIterAfterEpoch = (epoch+1) * \
                len(train_loader) * cfg_student.TRAIN.BATCH_SIZE
            for metricName, metricValue in metricsDict.items():
                logger.add_scalar('test/'+metricName,
                                  metricValue, global_step=sampleIterAfterEpoch)
            if isBetter:
                bestMetrics = metricsDict
                save_best_model(net_student, optmzr, work_dir, distributed)
                dist_print('Save best model: epoch %d' % epoch)
        # save_model(net, optimizer, epoch, work_dir, distributed)
    logger.close()