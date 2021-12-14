'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import os, random, math
import time
import glob
import numpy as np
import shutil

import torch

import logging
import argparse
import traceback
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import sys
sys.path.append(os.path.abspath(os.path.join("..", os.getcwd())))
from config import Config
from lib import *
import torch.distributed as dist
from utils import *
from utils.build import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Place config Congfile!')
parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--nprocs', type=int, default=1)

parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
parser.add_argument('--save_output', action='store_true', help='Save logits?')
parser.add_argument('--fp16', action='store_true', help='Training with fp16')
parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')

parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
parser.add_argument('--save', type=str, default='Checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()
args = Config(args)

#====================================================
# Some configuration
#====================================================

try:
    if args.resume:
        args.save = os.path.split(args.resume)[0]
    else:
        args.save = '{}/{}-EXP-{}'.format(args.save, args.Network, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=[args.config] + glob.glob('./tools/train*.py')+glob.glob('./lib/model/*.py'))
except:
    pass
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#---------------------------------
# Fusion Net Training
#---------------------------------
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

    # ---------------------------
    # Init distribution
    # ---------------------------
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')

    # ----------------------------
    # build function
    # ----------------------------
    model = build_model(args)
    model = model.cuda(local_rank)

    criterion = build_loss(args)
    optimizer = build_optim(args, model)
    scheduler = build_scheduler(args, optimizer)

    train_queue, train_sampler = build_dataset(args, phase='train')
    valid_queue, valid_sampler = build_dataset(args, phase='valid')

    if args.resume:
        model, optimizer, strat_epoch, best_acc = load_checkpoint(model, args.resume, optimizer)
        logging.info("The network will resume training.")
        logging.info("Start Epoch: {}, Learning rate: {}, Best accuracy: {}".format(strat_epoch, [g['lr'] for g in
                                                                                                  optimizer.param_groups],
                                                                                    round(best_acc, 4)))
        if args.resumelr:
            for g in optimizer.param_groups: g['lr'] = args.resumelr
            args.resume_scheduler = cosine_scheduler(args.resumelr, 1e-5, args.epochs - strat_epoch, len(train_queue))

    else:
        strat_epoch = 0
        best_acc = 0.0
    scheduler[0].last_epoch = strat_epoch


    if args.SYNC_BN and args.nprocs > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    if local_rank == 0:
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


    train_results = dict(
        train_score=[],
        train_loss=[],
        valid_score=[],
        valid_loss=[],
        best_score=0.0
    )
    if args.eval_only:
        valid_acc, _, _, meter_dict = infer(valid_queue, model, criterion, local_rank, 0)
        valid_acc = max(meter_dict['Acc_all'].avg, meter_dict['Acc'].avg, meter_dict['Acc_3'].avg)
        logging.info('valid_acc: {}, Acc_1: {}, Acc_2: {}, Acc_3: {}'.format(valid_acc, meter_dict['Acc_1'].avg, meter_dict['Acc_2'].avg, meter_dict['Acc_3'].avg))
        return

    #---------------------------
    # Mixed Precision Training
    # --------------------------
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    for epoch in range(strat_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if epoch < args.scheduler['warm_up_epochs']:
            for g in optimizer.param_groups:
                g['lr'] = scheduler[-1](epoch)

        args.epoch = epoch
        train_acc, train_obj, meter_dict_train = train(train_queue, model, criterion, optimizer, epoch, local_rank, scaler)
        valid_acc, valid_obj, valid_dict, meter_dict_val = infer(valid_queue, model, criterion, local_rank, epoch)
        valid_acc = max(meter_dict_val['Acc_all'].avg, meter_dict_val['Acc'].avg, meter_dict_val['Acc_3'].avg)
        if epoch >= args.scheduler['warm_up_epochs']:
            if args.scheduler['name'] == 'ReduceLR':
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
            logging.info('valid_acc: {}, Acc_1: {}, Acc_2: {}, Acc_3: {}, best acc: {}'.format(meter_dict_val['Acc'].avg, meter_dict_val['Acc_1'].avg,
                                                                                 meter_dict_val['Acc_2'].avg,
                                                                                 meter_dict_val['Acc_3'].avg, best_acc))

            state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'bestacc': best_acc}
            save_checkpoint(state, isbest, args.save)

            train_results['train_score'].append(train_acc)
            train_results['train_loss'].append(train_obj)
            train_results['valid_score'].append(valid_acc)
            train_results['valid_loss'].append(valid_obj)
            train_results['best_score'] = best_acc
            train_results.update(valid_dict)
            train_results['categories'] = np.unique(valid_dict['grounds'])

            if isbest:
                EvaluateMetric(PREDICTIONS_PATH=args.save, train_results=train_results, idx=epoch)
                for k, v in train_results.items():
                    if isinstance(v, list):
                        v.clear()

def train(train_queue, model, criterion, optimizer, epoch, local_rank, scaler):
    model.train()
    meter_dict = dict(
        Total_loss=AverageMeter(),
        MSE_loss=AverageMeter(),
        CE_loss=AverageMeter(),
        BCE_loss=AverageMeter(),
        Distill_loss = AverageMeter()

    )
    meter_dict['Data_Time'] = AverageMeter()
    meter_dict.update(dict(
        Acc_1=AverageMeter(),
        Acc_2=AverageMeter(),
        Acc_3=AverageMeter(),
        Acc=AverageMeter()
    ))

    end = time.time()
    for step, (inputs, heatmap, target, _) in enumerate(train_queue):
        meter_dict['Data_Time'].update((time.time() - end)/args.batch_size)
        inputs, target, heatmap = map(lambda x: [d.cuda(local_rank, non_blocking=True) for d in x] if isinstance(x, list) else x.cuda(local_rank, non_blocking=True), [inputs, target, heatmap])

        if args.resumelr:
            for g in optimizer.param_groups:
                 g['lr'] = args.resume_scheduler[len(train_queue) * args.resume_epoch + step]
        # ---------------------------
        # Mixed Precision Training
        # --------------------------
        if args.fp16:
            print('Train with FP16')
            optimizer.zero_grad()
            # Runs the forward pass with autocasting.
            with torch.cuda.amp.autocast():
                (logits, logit_r, logit_d), (CE_loss, BCE_loss, MSE_loss, distillation) = model(inputs, heatmap, target)
                globals()['CE_loss'] = CE_loss
                globals()['MSE_loss'] = MSE_loss
                globals()['BCE_loss'] = BCE_loss
                globals()['Distill_loss'] = distillation
                globals()['Total_loss'] = CE_loss + MSE_loss + BCE_loss + distillation

            scaler.scale(Total_loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # ---------------------------
            # Fp32 Precision Training
            # --------------------------
            (logits, logit_r, logit_d), (CE_loss, BCE_loss, MSE_loss, distillation) = model(inputs, heatmap, target)
            globals()['CE_loss'] = CE_loss
            globals()['MSE_loss'] = MSE_loss
            globals()['BCE_loss'] = BCE_loss
            globals()['Distill_loss'] = distillation
            globals()['Total_loss'] = CE_loss + MSE_loss + BCE_loss + distillation

            optimizer.zero_grad()
            Total_loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
            optimizer.step()

        #---------------------
        # Meter performance
        #---------------------
        torch.distributed.barrier()
        globals()['Acc'] = calculate_accuracy(logits, target)
        globals()['Acc_1'] = calculate_accuracy(logit_r, target)
        globals()['Acc_2'] = calculate_accuracy(logit_d, target)
        globals()['Acc_3'] = calculate_accuracy(logit_r+logit_d, target)

        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
            if 'Acc' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))

        if step % args.report_freq == 0 and local_rank == 0:

            log_info = {
                'Epoch': '{}/{}'.format(epoch + 1, args.epochs),
                'Mini-Batch': '{:0>5d}/{:0>5d}'.format(step + 1,
                                                       len(train_queue.dataset) // (args.batch_size * args.nprocs)),
                'Lr': ['{:.4f}'.format(g['lr'])  for g in optimizer.param_groups],
            }
            log_info.update(dict((name, '{:.4f}'.format(value.avg)) for name, value in meter_dict.items()))
            print_func(log_info)
        end = time.time()
    args.resume_epoch += 1
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
        MSE_loss=AverageMeter(),
        CE_loss=AverageMeter(),
        Distill_loss=AverageMeter()
    )
    meter_dict.update(dict(
        Acc_1=AverageMeter(),
        Acc_2=AverageMeter(),
        Acc_3=AverageMeter(),
        Acc = AverageMeter(),
        Acc_all=AverageMeter(),
    ))

    meter_dict['Infer_Time'] = AverageMeter()
    grounds, preds, v_paths = [], [], []
    for step, (inputs, heatmap, target, v_path) in enumerate(valid_queue):
        end = time.time()
        inputs, target, heatmap = map(
            lambda x: [d.cuda(local_rank, non_blocking=True) for d in x] if isinstance(x, list) else x.cuda(local_rank,
                                                                                                            non_blocking=True),
            [inputs, target, heatmap])
        if args.fp16:
            with torch.cuda.amp.autocast():
                (logits, logit_r, logit_d), (CE_loss, BCE_loss, MSE_loss, distillation) = model(inputs, heatmap, target)
        else:
            (logits, logit_r, logit_d), (CE_loss, BCE_loss, MSE_loss, distillation) = model(inputs, heatmap, target)
        meter_dict['Infer_Time'].update((time.time() - end) / args.test_batch_size)
        globals()['CE_loss'] = CE_loss
        globals()['MSE_loss'] = MSE_loss
        globals()['BCE_loss'] = BCE_loss
        globals()['Distill_loss'] = distillation
        globals()['Total_loss'] = CE_loss + MSE_loss + BCE_loss + distillation

        torch.distributed.barrier()
        globals()['Acc'] = calculate_accuracy(logits, target)
        globals()['Acc_1'] = calculate_accuracy(logit_r, target)
        globals()['Acc_2'] = calculate_accuracy(logit_d, target)
        globals()['Acc_3'] = calculate_accuracy(logit_r+logit_d, target)
        globals()['Acc_all'] = calculate_accuracy(logit_r+logit_d+logits, target)


        grounds += target.cpu().tolist()
        preds += torch.argmax(logits, dim=1).cpu().tolist()
        v_paths += v_path
        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
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

    torch.distributed.barrier()
    grounds_gather = concat_all_gather(torch.tensor(grounds).cuda(local_rank))
    preds_gather = concat_all_gather(torch.tensor(preds).cuda(local_rank))
    grounds_gather, preds_gather = list(map(lambda x: x.cpu().numpy(), [grounds_gather, preds_gather]))

    if local_rank == 0:
        v_paths = np.array(v_paths)
        grounds = np.array(grounds)
        preds = np.array(preds)
        wrong_idx = np.where(grounds != preds)
        v_paths = v_paths[wrong_idx[0]]
        grounds = grounds[wrong_idx[0]]
        preds = preds[wrong_idx[0]]
    return meter_dict['Acc'].avg, meter_dict['Total_loss'].avg, dict(grounds=grounds_gather, preds=preds_gather, valid_images=(v_paths, grounds, preds)), meter_dict

if __name__ == '__main__':
    try:
        main(args.local_rank, args.nprocs, args)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove ‘{}’: Directory'.format(args.save))
            os.system('rm -rf {} \n mv {} ./Checkpoints/trash'.format(args.save, args.save))
        os._exit(0)
    except Exception:
        print(traceback.print_exc())
        if os.path.exists(args.save) and len(os.listdir(args.save)) < 3:
            print('remove ‘{}’: Directory'.format(args.save))
            os.system('rm -rf {} \n mv {} ./Checkpoints/trash'.format(args.save, args.save))
        os._exit(0)
    finally:
        torch.cuda.empty_cache()
