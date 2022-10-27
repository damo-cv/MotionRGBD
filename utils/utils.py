'''
This file is modified from:
https://github.com/yuhuixu1993/PC-DARTS/blob/master/utils.py
'''

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict

class ClassAcc():
    def __init__(self, GESTURE_CLASSES):
        self.class_acc = dict(zip([i for i in range(GESTURE_CLASSES)], [0]*GESTURE_CLASSES))
        self.single_class_num = [0]*GESTURE_CLASSES
    def update(self, logits, target):
        pred = torch.argmax(logits, dim=1)
        for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
            if p == t:
                self.class_acc[t] += 1
            self.single_class_num[t] += 1
    def result(self):
        return [round(v / (self.single_class_num[k]+0.000000001), 4) for k, v in self.class_acc.items()]
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def adjust_learning_rate(optimizer, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    df = 0.7
    ds = 40000.0
    lr = lr * np.power(df, step / ds)
    # lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(1, 1, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        #n_correct_elems = correct.float().sum().data[0]
        # n_correct_elems = correct.float().sum().item()
    # return n_correct_elems / batch_size
    return correct_k.mul_(1.0 / batch_size)

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best=False, save='./', filename='checkpoint.pth.tar'):
  filename = os.path.join(save, filename)
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def load_checkpoint(model, model_path, optimizer=None):
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(4))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    bestacc = checkpoint['bestacc']
    return model, optimizer, epoch, bestacc

def load_pretrained_checkpoint(model, model_path):
    # params = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(local_rank))['model']
    params = torch.load(model_path, map_location='cpu')['model']
    new_state_dict = OrderedDict()

    for k, v in params.items():
        name = k[7:] if k[:7] == 'module.' else k
        try:
            if v.shape == model.state_dict()[name].shape:
              if name not in ['dtn.mlp_head_small.1.bias', "dtn.mlp_head_small.1.weight",
                        'dtn.mlp_head_media.1.bias', "dtn.mlp_head_media.1.weight",
                        'dtn.mlp_head_large.1.bias', "dtn.mlp_head_large.1.weight"]:
                  new_state_dict[name] = v
        except:
            continue
    ret = model.load_state_dict(new_state_dict, strict=False)
    print('Missing keys: \n', ret.missing_keys)
    return model

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      if os.path.isdir(script) and script != '__pycache__':
        dst_file = os.path.join(path, 'scripts', script)
        shutil.copytree(script, dst_file)
      else:
        dst_file = os.path.join(path, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule