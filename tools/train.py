'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import time
import glob
import numpy as np
import shutil
import cv2
import os, random, math
import sys
sys.path.append(os.path.join('..', os.path.abspath(os.path.join(os.getcwd()))) )

import torch
import utils
import logging
import argparse
import traceback
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# import flops_benchmark
from utils.visualizer import Visualizer
from config import Config
from lib import *
from utils import *

#------------------------
# evaluation metrics
#------------------------
from sklearn.decomposition import PCA
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt  # For graphics
import seaborn as sns
from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Load Congfile.')
parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--nprocs', type=int, default=1)

parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
parser.add_argument('--save_output', action='store_true', help='Save logits?')
parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')
parser.add_argument('--resume', type=str, default='', help='resume model path.')

parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
parser.add_argument('--save', type=str, default='Checkpoints/', help='experiment dir')
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()
args = Config(args)

try:
    if args.resume:
        args.save = os.path.split(args.resume)[0]
    else:
        args.save = '{}/{}-{}-{}-{}'.format(args.save, args.Network, args.dataset, args.type, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=[args.config] + glob.glob('./tools/*.py') + glob.glob('./lib/*'))
except:
    pass
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt.item()

def main(local_rank, nprocs, args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % local_rank)

    #---------------------------
    # Init distribution
    #---------------------------
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')

    #----------------------------
    # build function
    #----------------------------
    model = build_model(args)
    model = model.cuda(local_rank)

    criterion = build_loss(args)
    optimizer = build_optim(args, model)
    scheduler = build_scheduler(args, optimizer)

    train_queue, train_sampler = build_dataset(args, phase='train')
    valid_queue, valid_sampler = build_dataset(args, phase='valid')


    if args.resume:
        model, optimizer, strat_epoch, best_acc = load_checkpoint(model, args.resume, optimizer)
        logging.info("Start Epoch: {}, Learning rate: {}, Best accuracy: {}".format(strat_epoch, [g['lr'] for g in
                                                                                                  optimizer.param_groups],
                                                                                    round(best_acc, 4)))
        if args.resumelr:
            for g in optimizer.param_groups:
                args.resumelr = g['lr'] if not isinstance(args.resumelr, float) else args.resumelr
                g['lr'] = args.resumelr
            #resume_scheduler = np.linspace(args.resumelr, 1e-5, args.epochs - strat_epoch)
            resume_scheduler = cosine_scheduler(args.resumelr, 1e-5, args.epochs - strat_epoch + 1, niter_per_ep=1).tolist()
            resume_scheduler.pop(0)

        args.epoch = strat_epoch - 1
    else:
        strat_epoch = 0
        best_acc = 0.0
        args.epoch = strat_epoch

    scheduler[0].last_epoch = strat_epoch

    if args.SYNC_BN and args.nprocs > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    if local_rank == 0:
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        # logging.info('FLOPs: {}'.format(flops_benchmark.count_flops(model)))

    train_results = dict(
        train_score=[],
        train_loss=[],
        valid_score=[],
        valid_loss=[],
        best_score=0.0
    )
    if args.eval_only:
        valid_acc, _, _, meter_dict, output = infer(valid_queue, model, criterion, local_rank, strat_epoch)
        logging.info('valid_acc: {}, Acc_1: {}, Acc_2: {}, Acc_3: {}'.format(valid_acc, meter_dict['Acc_1'].avg, meter_dict['Acc_2'].avg, meter_dict['Acc_3'].avg))
        if args.save_output:
            torch.save(output, os.path.join(args.save, '{}-output.pth'.format(args.type)))
        return

    for epoch in range(strat_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if epoch < args.scheduler['warm_up_epochs']-1:
            args.distill_lamdb = 0.
            for g in optimizer.param_groups:
                g['lr'] = scheduler[-1](epoch)
        else:
            args.distill_lamdb = args.distill

        args.epoch = epoch
        train_acc, train_obj, meter_dict_train = train(train_queue, model, criterion, optimizer, epoch, local_rank)
        valid_acc, valid_obj, valid_dict, meter_dict_val, output = infer(valid_queue, model, criterion, local_rank, epoch)

        # scheduler_func.step(scheduler, valid_acc)
        if epoch >= args.scheduler['warm_up_epochs']:
            if args.resume and args.resumelr:
                for g in optimizer.param_groups:
                    g['lr'] = resume_scheduler[0]
                resume_scheduler.pop(0)
            elif args.scheduler['name'] == 'ReduceLR':
                scheduler[0].step(valid_acc)
            else:
                scheduler[0].step()

        if local_rank == 0:
            if valid_acc > best_acc:
                best_acc = valid_acc
                isbest = True
            else:
                isbest = False
            logging.info('train_acc %f', train_acc)
            logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
            state = {'model': model.module.state_dict(),'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'bestacc': best_acc}
            save_checkpoint(state, isbest, args.save)

            train_results['train_score'].append(train_acc)
            train_results['train_loss'].append(train_obj)
            train_results['valid_score'].append(valid_acc)
            train_results['valid_loss'].append(valid_obj)
            train_results['best_score'] = best_acc
            train_results.update(valid_dict)
            train_results['categories'] = np.unique(valid_dict['grounds'])

            if args.visdom['enable']:
                vis.plot_many({'train_acc': train_acc, 'loss': train_obj,
                               'cosin_similar': meter_dict_train['cosin_similar'].avg}, 'Train-' + args.type, epoch)
                vis.plot_many({'valid_acc': valid_acc, 'loss': valid_obj,
                               'cosin_similar': meter_dict_val['cosin_similar'].avg}, 'Valid-' + args.type, epoch)

            if isbest:
                if args.save_output:
                    torch.save(output, os.path.join(args.save, '{}-output.pth'.format(args.type)))
                EvaluateMetric(PREDICTIONS_PATH=args.save, train_results=train_results, idx=epoch)
                for k, v in train_results.items():
                    if isinstance(v, list):
                        v.clear()

def Visfeature(inputs, feature, v_path=None):
    if args.visdom['enable']:
        vis.featuremap('CNNVision',
                       torch.sum(make_grid(feature[0].detach(), nrow=int(feature[0].size(0) ** 0.5), padding=2), dim=0).flipud())
        vis.featuremap('Attention Maps Similarity',
                       make_grid(feature[1], nrow=int(feature[1].detach().cpu().size(0) ** 0.5), padding=2)[0].flipud())

        vis.featuremap('Enhancement Weights', feature[3].flipud())
    else:
        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(
            torch.sum(make_grid(feature[0].detach(), nrow=int(feature[0].size(0) ** 0.5), padding=2), dim=0).cpu().numpy(),
            annot=False, fmt='g', ax=ax)
        ax.set_title('CNNVision', fontsize=10)
        fig.savefig(os.path.join(args.save, 'CNNVision.jpg'), dpi=fig.dpi)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(make_grid(feature[1].detach(), nrow=int(feature[1].size(0) ** 0.5), padding=2)[0].cpu().numpy(), annot=False,
                    fmt='g', ax=ax)
        ax.set_title('Attention Maps Similarity', fontsize=10)
        fig.savefig(os.path.join(args.save, 'AttMapSimilarity.jpg'), dpi=fig.dpi)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(feature[3].detach().cpu().numpy(), annot=False, fmt='g', ax=ax)
        ax.set_title('Enhancement Weights', fontsize=10)
        fig.savefig(os.path.join(args.save, 'EnhancementWeights.jpg'), dpi=fig.dpi)
        plt.close()

    #------------------------------------------
    # Spatial feature visualization
    #------------------------------------------
    headmap = feature[-1][0][0,:].detach().cpu().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)  # torch.Size([64, 7, 7])
    headmap = torch.from_numpy(headmap)
    img = feature[-1][1]

    result = []
    for map, mg in zip(headmap.unsqueeze(1), img.permute(1,2,3,0)):
        map = cv2.resize(map.squeeze().cpu().numpy(), (mg.shape[0]//2, mg.shape[1]//2))
        map = np.uint8(255 * map)
        map = cv2.applyColorMap(map, cv2.COLORMAP_JET)

        mg = np.uint8(mg.cpu().numpy() * 128 + 127.5)
        mg = cv2.resize(mg, (mg.shape[0]//2, mg.shape[1]//2))
        superimposed_img = cv2.addWeighted(mg, 0.4, map, 0.6, 0)

        result.append(torch.from_numpy(superimposed_img).unsqueeze(0))
    superimposed_imgs = torch.cat(result).permute(0, 3, 1, 2)
    # save_image(superimposed_imgs, os.path.join(args.save, 'CAM-Features.png'), nrow=int(superimposed_imgs.size(0) ** 0.5), padding=2).permute(1,2,0)
    superimposed_imgs = make_grid(superimposed_imgs, nrow=int(superimposed_imgs.size(0) ** 0.5), padding=2).permute(1,2,0)
    cv2.imwrite(os.path.join(args.save, 'CAM-Features.png'), superimposed_imgs.numpy())

    if args.eval_only:
        MHAS_s, MHAS_m, MHAS_l = feature[2]
        MHAS_s, MHAS_m, MHAS_l = MHAS_s.detach().cpu(), MHAS_m.detach().cpu(), MHAS_l.detach().cpu()
        # Normalize
        att_max, index_max = torch.max(MHAS_s.view(MHAS_s.size(0), -1), dim=-1)
        att_min, index_min = torch.min(MHAS_s.view(MHAS_s.size(0), -1), dim=-1)
        MHAS_s = (MHAS_s - att_min.view(-1, 1, 1))/(att_max.view(-1, 1, 1) - att_min.view(-1, 1, 1))

        att_max, index_max = torch.max(MHAS_m.view(MHAS_m.size(0), -1), dim=-1)
        att_min, index_min = torch.min(MHAS_m.view(MHAS_m.size(0), -1), dim=-1)
        MHAS_m = (MHAS_m - att_min.view(-1, 1, 1))/(att_max.view(-1, 1, 1) - att_min.view(-1, 1, 1))

        att_max, index_max = torch.max(MHAS_l.view(MHAS_l.size(0), -1), dim=-1)
        att_min, index_min = torch.min(MHAS_l.view(MHAS_l.size(0), -1), dim=-1)
        MHAS_l = (MHAS_l - att_min.view(-1, 1, 1))/(att_max.view(-1, 1, 1) - att_min.view(-1, 1, 1))

        mhas_s = make_grid(MHAS_s.unsqueeze(1), nrow=int(MHAS_s.size(0) ** 0.5), padding=2)[0]
        mhas_m = make_grid(MHAS_m.unsqueeze(1), nrow=int(MHAS_m.size(0) ** 0.5), padding=2)[0]
        mhas_l = make_grid(MHAS_l.unsqueeze(1), nrow=int(MHAS_l.size(0) ** 0.5), padding=2)[0]
        if args.visdom['enable']:
            vis.featuremap('MHAS Map', mhas_l)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(131)
        sns.heatmap(mhas_s.squeeze(), annot=False, fmt='g', ax=ax)
        ax.set_title('\nMHSA Small', fontsize=10)

        ax = fig.add_subplot(132)
        sns.heatmap(mhas_m.squeeze(), annot=False, fmt='g', ax=ax)
        ax.set_title('\nMHSA Medium', fontsize=10)

        ax = fig.add_subplot(133)
        sns.heatmap(mhas_l.squeeze(), annot=False, fmt='g', ax=ax)
        ax.set_title('\nMHSA Large', fontsize=10)
        plt.suptitle('{}'.format(v_path[0].split('/')[-1]), fontsize=20)
        fig.savefig('demo/{}-MHAS.jpg'.format(args.save.split('/')[-1]), dpi=fig.dpi)
        plt.close()

def train(train_queue, model, criterion, optimizer, epoch, local_rank):
    model.train()

    meter_dict = dict(
        Total_loss=AverageMeter(),
        CE_loss=AverageMeter(),
        Distil_loss=AverageMeter()
    )
    meter_dict.update(dict(
        cosin_similar=AverageMeter()
    ))
    meter_dict['Data_Time'] = AverageMeter()
    meter_dict.update(dict(
        Acc_1=AverageMeter(),
        Acc_2=AverageMeter(),
        Acc_3=AverageMeter(),
        Acc=AverageMeter()
    ))

    end = time.time()
    CE = criterion
    for step, (inputs, heatmap, target, _) in enumerate(train_queue):
        meter_dict['Data_Time'].update((time.time() - end)/args.batch_size)
        inputs, target, heatmap = map(lambda x: x.cuda(local_rank, non_blocking=True), [inputs, target, heatmap])

        (logits, xs, xm, xl), distillation_loss, feature = model(inputs, heatmap)
        if args.MultiLoss:
            lamd1, lamd2, lamd3, lamd4 = map(float, args.loss_lamdb)
            globals()['CE_loss'] = lamd1*CE(logits, target) + lamd2*CE(xs, target) + lamd3*CE(xm, target) + lamd4*CE(xl, target)
        else:
            globals()['CE_loss'] = CE(logits, target)
        globals()['Distil_loss'] = distillation_loss * args.distill_lamdb
        globals()['Total_loss'] = CE_loss + Distil_loss

        optimizer.zero_grad()
        Total_loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        #---------------------
        # Meter performance
        #---------------------
        torch.distributed.barrier()
        globals()['Acc'] = calculate_accuracy(logits, target)
        globals()['Acc_1'] = calculate_accuracy(xs, target)
        globals()['Acc_2'] = calculate_accuracy(xm, target)
        globals()['Acc_3'] = calculate_accuracy(xl, target)

        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
            if 'cosin' in name:
                meter_dict[name].update(float(feature[2]))
            if 'Acc' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))

        if (step+1) % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': '{}/{}'.format(epoch + 1, args.epochs),
                'Mini-Batch': '{:0>5d}/{:0>5d}'.format(step + 1,
                                                       len(train_queue.dataset) // (args.batch_size * args.nprocs)),
                'Lr': [round(float(g['lr']), 7) for g in optimizer.param_groups],
            }
            log_info.update(dict((name, '{:.4f}'.format(value.avg)) for name, value in meter_dict.items()))
            print_func(log_info)

            if args.vis_feature:
                Visfeature(inputs, feature)
        end = time.time()

    return meter_dict['Acc'].avg, meter_dict['Total_loss'].avg, meter_dict

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def infer(valid_queue, model, criterion, local_rank, epoch):
    model.eval()

    meter_dict = dict(
        Total_loss=AverageMeter(),
        CE_loss=AverageMeter(),
        Distil_loss=AverageMeter()
    )
    meter_dict.update(dict(
        cosin_similar=AverageMeter(),
    ))
    meter_dict.update(dict(
        Acc_1=AverageMeter(),
        Acc_2=AverageMeter(),
        Acc_3=AverageMeter(),
        Acc=AverageMeter()
    ))

    meter_dict['Infer_Time'] = AverageMeter()
    CE = criterion
    grounds, preds, v_paths = [], [], []
    output = {}
    for step, (inputs, heatmap, target, v_path) in enumerate(valid_queue):
        n = inputs.size(0)
        end = time.time()
        inputs, target, heatmap = map(lambda x: x.cuda(local_rank, non_blocking=True), [inputs, target, heatmap])

        (xs, xm, xl, logits), distillation_loss, feature = model(inputs, heatmap)
        meter_dict['Infer_Time'].update((time.time() - end) / n)

        if args.MultiLoss:
            lamd1, lamd2, lamd3, lamd4 = map(float, args.loss_lamdb)
            globals()['CE_loss'] = lamd1 * CE(logits, target) + lamd2 * CE(xs, target) + lamd3 * CE(xm,
                                                                                                    target) + lamd4 * CE(
                xl, target)
        else:
            globals()['CE_loss'] = CE(logits, target)
        globals()['Distil_loss'] = distillation_loss * args.distill_lamdb
        globals()['Total_loss'] = CE_loss + Distil_loss

        grounds += target.cpu().tolist()
        preds += torch.argmax(logits, dim=1).cpu().tolist()
        v_paths += v_path
        torch.distributed.barrier()
        globals()['Acc'] = calculate_accuracy(logits, target)
        globals()['Acc_1'] = calculate_accuracy(xs+xm, target)
        globals()['Acc_2'] = calculate_accuracy(xs+xl, target)
        globals()['Acc_3'] = calculate_accuracy(xl+xm, target)

        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
            if 'cosin' in name:
                meter_dict[name].update(float(feature[2]))
            if 'Acc' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))

        if step % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': epoch + 1,
                'Mini-Batch': '{:0>4d}/{:0>4d}'.format(step + 1, len(valid_queue.dataset) // (
                            args.test_batch_size * args.nprocs)),
            }
            log_info.update(dict((name, '{:.4f}'.format(value.avg)) for name, value in meter_dict.items()))
            print_func(log_info)
            if args.vis_feature:
                Visfeature(inputs, feature, v_path)

        if args.save_output:
            for t, logit in zip(v_path, logits):
                output[t] = logit
    torch.distributed.barrier()
    grounds_gather = concat_all_gather(torch.tensor(grounds).cuda(local_rank))
    preds_gather = concat_all_gather(torch.tensor(preds).cuda(local_rank))
    grounds_gather, preds_gather = list(map(lambda x: x.cpu().numpy(), [grounds_gather, preds_gather]))

    if local_rank == 0:
        # v_paths = np.array(v_paths)[random.sample(list(wrong), 10)]
        v_paths = np.array(v_paths)
        grounds = np.array(grounds)
        preds = np.array(preds)
        wrong_idx = np.where(grounds != preds)
        v_paths = v_paths[wrong_idx[0]]
        grounds = grounds[wrong_idx[0]]
        preds = preds[wrong_idx[0]]
    return max(meter_dict['Acc'].avg, meter_dict['Acc_1'].avg, meter_dict['Acc_2'].avg, meter_dict['Acc_3'].avg), meter_dict['Total_loss'].avg, dict(grounds=grounds_gather, preds=preds_gather, valid_images=(v_paths, grounds, preds)), meter_dict, output

if __name__ == '__main__':
    if args.visdom['enable']:
        vis = Visualizer(args.visdom['visname'])
    try:
        main(args.local_rank, args.nprocs, args)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove {}: Directory'.format(args.save))
            os.system('rm -rf {} \n mv {} ./Checkpoints/trash'.format(args.save, args.save))
        os._exit(0)
    except Exception:
        print(traceback.print_exc())
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove {}: Directory'.format(args.save))
            os.system('rm -rf {} \n mv {} ./Checkpoints/trash'.format(args.save, args.save))
        os._exit(0)
    finally:
        torch.cuda.empty_cache()
