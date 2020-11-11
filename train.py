import torch
import os
import datetime
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, save_best_model, cp_projects
from utils.common import get_work_dir, get_logger

from test_wrapper import testNet
from utils.visualize import genSegLabelImage, logSegLabelImage
from data.constant import tusimple_row_anchor, culane_row_anchor

import time


def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.cuda(), seg_label.cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}
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


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux, cfg, cls_num_per_lane):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)

        global_batch_step = (epoch * len(data_loader) + b_idx)
        global_sample_iter = global_batch_step * cfg.batch_size

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        if global_batch_step % 200 == 0:
            if cfg.dataset == 'CULane':
                cls_num_per_lane = 18
                img_w, img_h = 1640, 590
                row_anchor = culane_row_anchor
            elif cfg.dataset == 'Tusimple':
                cls_num_per_lane = 56
                img_w, img_h = 1280, 720
                row_anchor = tusimple_row_anchor

            logSegLabelImage(logger,
                             'predImg',
                             global_sample_iter,
                             data_label[0][0],
                             results['cls_out'][0],
                             row_anchor,
                             cfg.griding_num,
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
        t_data_0 = time.time()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime(
        '[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101',
                            '152', '50next', '101next', '50wide', '101wide']

    train_loader, cls_num_per_lane = get_train_loader(
        cfg.batch_size, cfg.data_root,
        cfg.griding_num, cfg.dataset,
        cfg.use_aux, distributed,
        cfg.num_lanes, cfg.train_ds_proportion)

    net = parsingNet(pretrained=True, backbone=cfg.backbone, cls_dim=(
        cfg.griding_num+1, cls_num_per_lane, cfg.num_lanes), use_aux=cfg.use_aux, use_spp=cfg.use_spp).cuda()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(
        optimizer, cfg, len(train_loader) * cfg.batch_size)
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(work_dir)

    bestMetrics = None
    logger.add_text('configuration', str(cfg))

    for epoch in range(resume_epoch, cfg.epoch):

        train(net, train_loader, loss_dict, optimizer, scheduler,
              logger, epoch, metric_dict, cfg.use_aux, cfg, cls_num_per_lane)
        if cfg.test_during_train and (epoch % cfg.test_interval == 0):
            metricsDict, isBetter = testNet(
                net, args, cfg, True, lastMetrics=bestMetrics)
            sampleIterAfterEpoch = (epoch+1) * \
                len(train_loader) * cfg.batch_size
            for metricName, metricValue in metricsDict.items():
                logger.add_scalar('test/'+metricName,
                                  metricValue, global_step=sampleIterAfterEpoch)
            if isBetter:
                bestMetrics = metricsDict
                save_best_model(net, optimizer, work_dir, distributed)
                dist_print('Save best model: epoch %d' % epoch)
        # save_model(net, optimizer, epoch, work_dir, distributed)
    logger.close()
