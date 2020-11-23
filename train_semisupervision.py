import torch
import os
import datetime
import numpy as np

from model.model import parsingNet
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


def calc_loss(loss_dict, results, logger, global_step, trainIdentider):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar(
                'loss_%s/%s_%s' % (loss_dict['name'][i], loss_dict['name'][i], trainIdentider), loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux, cfg, trainIdentider):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)

        global_batch_step = (epoch * len(data_loader) + b_idx)
        global_sample_iter = global_batch_step * cfg.TRAIN.BATCH_SIZE

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux, load_name=True)

        if global_batch_step % 200 == 0:
            cls_num_per_lane = cfg.NETWORK.CLS_NUM_PER_LANE
            img_w, img_h = cfg.DATASET.RAW_IMG_SIZE
            if cfg.DATASET.NAME == 'CULane':
                row_anchor = culane_row_anchor
            elif cfg.DATASET.NAME == 'Tusimple':
                row_anchor = tusimple_row_anchor

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

        loss = calc_loss(loss_dict, results, logger,
                         global_sample_iter, trainIdentider)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_sample_iter)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)

        logger.add_scalar(
            'meta_epoch/epoch_%s' % trainIdentider, epoch, global_sample_iter)

        if global_batch_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric_%s/%s_%s' % (me_name, me_name, trainIdentider),
                                  me_op.get(), global_step=global_sample_iter)
        logger.add_scalar(
            'meta_lr/lr_%s' % trainIdentider, optimizer.param_groups[0]['lr'], global_step=global_sample_iter)

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


def train_proc(net, optimizer, scheduler, train_loader, args, cfg, logger, bestMetrics, resume_epoch, trainIdentider):
    for epoch in range(resume_epoch, cfg.TRAIN.EPOCH):
        train(net, train_loader, loss_dict, optimizer, scheduler,
              logger, epoch, metric_dict, cfg.NETWORK.USE_AUX, cfg, trainIdentider)
        if cfg.TEST.DURING_TRAIN and (epoch % cfg.TEST.INTERVAL == 0):
            metricsDict, isBetter = testNet(
                net, args, cfg, True, lastMetrics=bestMetrics)
            sampleIterAfterEpoch = (epoch+1) * \
                len(train_loader) * cfg.TRAIN.BATCH_SIZE
            for metricName, metricValue in metricsDict.items():
                logger.add_scalar('test_%s/%s_%s' % (metricName, metricName, trainIdentider),
                                  metricValue, global_step=sampleIterAfterEpoch)
            if isBetter:
                bestMetrics = metricsDict
                save_best_model(net, optimizer, work_dir, distributed)
                dist_print('Save best model: trainId: %s, epoch %d' %
                           (trainIdentider, epoch))
        # save_model(net, optimizer, epoch, work_dir, distributed)
    return bestMetrics, sampleIterAfterEpoch


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
        resume_epoch = int(os.path.split(cfg.EXP.RESUME)[1][2:5]) + 1
    else:
        resume_epoch = 0

    return resume_epoch


def getVariousLoader(args, cfg):
    train_loaders, _ = get_train_loader(
        cfg.TRAIN.BATCH_SIZE, cfg.DATASET.ROOT,
        cfg.NETWORK.GRIDING_NUM, cfg.DATASET.NAME,
        cfg.NETWORK.USE_AUX, distributed,
        cfg.DATASET.NUM_LANES, cfg.DATASET.TRAIN_PROPORTION, split=True,
        split_proportion=cfg.DATASET.SEMI_SUPERVISION_SPLIT,
        load_name=True, pin_memory=cfg.DATASET.PIN_MEMORY
    )

    train_loaders_no_aug, _ = get_train_loader(
        cfg.TRAIN.BATCH_SIZE, cfg.DATASET.ROOT,
        cfg.NETWORK.GRIDING_NUM, cfg.DATASET.NAME+'-no-aug',
        cfg.NETWORK.USE_AUX, distributed,
        cfg.DATASET.NUM_LANES, cfg.DATASET.TRAIN_PROPORTION, split=True,
        split_proportion=cfg.DATASET.SEMI_SUPERVISION_SPLIT,
        load_name=True, pin_memory=cfg.DATASET.PIN_MEMORY
    )

    annotated_loader = train_loaders[0]
    pseudo_gen_loader = train_loaders_no_aug[1]

    return annotated_loader, pseudo_gen_loader


def getPseudoAnnotatedLoader(args, cfg):
    pseudo_annotated_loader, _ = get_train_loader(
        cfg.TRAIN.BATCH_SIZE, cfg.DATASET.ROOT,
        cfg.NETWORK.GRIDING_NUM, cfg.DATASET.NAME+"-pseudo",
        cfg.NETWORK.USE_AUX, distributed,
        cfg.DATASET.NUM_LANES, load_name=True, pin_memory=cfg.DATASET.PIN_MEMORY
    )
    return pseudo_annotated_loader


def getOptimizerAndSchedulerAndResumeEpoch(paramSet, net, loader, cfg):
    optimizer = get_optimizer(net, cfg, paramSet=paramSet)
    resume_epoch = recoveryState(net, optimizer, cfg)
    scheduler = get_scheduler(
        optimizer, cfg, len(loader) * cfg.TRAIN.BATCH_SIZE)
    return optimizer, scheduler, resume_epoch


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_yacs_config()

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
    assert cfg.NETWORK.BACKBONE in ['res18', 'res34', 'res50', 'res101',
                                    'res152', '50next', '101next', '50wide', '101wide']

    cls_num_per_lane = cfg.NETWORK.CLS_NUM_PER_LANE

    annotated_loader, pseudo_gen_loader = getVariousLoader(
        args, cfg)

    net_teacher = parsingNet(pretrained=True, backbone=cfg.NETWORK.BACKBONE, cls_dim=(
        cfg.NETWORK.GRIDING_NUM+1, cls_num_per_lane, cfg.DATASET.NUM_LANES), use_aux=cfg.NETWORK.USE_AUX, use_spp=cfg.NETWORK.USE_SPP)
    net_student = parsingNet(pretrained=True, backbone=cfg.NETWORK.BACKBONE, cls_dim=(
        cfg.NETWORK.GRIDING_NUM+1, cls_num_per_lane, cfg.DATASET.NUM_LANES), use_aux=cfg.NETWORK.USE_AUX, use_spp=cfg.NETWORK.USE_SPP)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])

    # Step 0: Train a teacher net with mannually annotated sample
    dist_print(
        'Iteration 0 Step 0: Train a teacher net with mannually annotated sample')

    dist_print(len(annotated_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(work_dir)

    bestMetrics = None
    logger.add_text('configuration', str(cfg))

    net_teacher = net_teacher.cuda()
    optmzr, scdulr, resume_epoch = getOptimizerAndSchedulerAndResumeEpoch(
        'TRAIN', net_teacher, annotated_loader, cfg)
    bestMetrics, _ = train_proc(net_teacher, optmzr, scdulr, annotated_loader,
                                args, cfg, logger, bestMetrics, resume_epoch, '0_S0')

    for metricName, metricValue in bestMetrics.items():
        logger.add_scalar('test_summary/%s' % metricName,
                          metricValue, global_step=0)

    for grandIterNum in range(1, 20):
        # Step1: Generate pseudo gt from teacher network
        dist_print(
            'Iteration %d Step 1: Generate pseudo gt from teacher network' % grandIterNum)
        net_teacher = net_teacher.cuda()
        genPseudoGt(net_teacher, pseudo_gen_loader, cfg.DATASET.ROOT,
                    "train_pseudo_gt.txt", "pseudo_clips_gt", grandIterNum)
        pseudo_annotated_loader = getPseudoAnnotatedLoader(args, cfg)
        net_teacher = net_teacher.cpu()

        dist_print(
            'Iteration %d Step 2: Train student network with pseduo gt' % grandIterNum)
        # Step2: Train student network with pseduo gt
        net_student = net_student.cuda()
        optmzr, scdulr, resume_epoch = getOptimizerAndSchedulerAndResumeEpoch(
            'TRAIN_PSEUDO', net_student, pseudo_annotated_loader, cfg)
        bestMetrics, _ = train_proc(net_student, optmzr, scdulr, pseudo_annotated_loader,
                                    args, cfg, logger, bestMetrics, resume_epoch, '%d_S2' % grandIterNum)

        # Step3: Finetune student network with mannually annotated sample
        dist_print(
            'Iteration %d Step 3: Finetune student network with mannually annotated sample' % grandIterNum)
        optmzr, scdulr, resume_epoch = getOptimizerAndSchedulerAndResumeEpoch(
            'TRAIN_FINETUNE', net_student, annotated_loader, cfg)
        bestMetrics, _ = train_proc(net_student, optmzr, scdulr, annotated_loader,
                                    args, cfg, logger, bestMetrics, resume_epoch, '%d_S3' % grandIterNum)
        net_student = net_student.cpu()

        for metricName, metricValue in bestMetrics.items():
            logger.add_scalar('test_summary/%s' % metricName,
                              metricValue, global_step=grandIterNum)
        # Step4: Exchange network
        dist_print(
            'Iteration %d Step 4: Exchange weights of teacher network and student network' % grandIterNum)
        net_teacher, net_student = net_student, net_teacher

    logger.close()
