import torch
import torch.nn as nn
import torch.optim as optim
from args import args
import torch.backends.cudnn as cudnn
import os
import time
import copy
import sys
import random
import numpy as np
import math
import heapq
from data import cifar10
from utils.logging import *
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing
from utils.schedulers import get_policy
from utils.conv_type import SparseConv
from importlib import import_module

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter

import models
import pdb
import datetime

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()

def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
ensure_path(log_base_dir)
logger = get_logger(os.path.join(log_base_dir, 'logger'+now+'.log'))
device = torch.device(f"cuda:{args.multigpu[0]}") if torch.cuda.is_available() else 'cpu'

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# Data
print('==> Loading Data..')
loader = cifar10.Cifar10(args)

def set_model(model, pr_target):
    masks = []
    for n, m in model.named_modules():
        if hasattr(m, 'mask'):
            mask = m.mask.view(-1).clone()
            masks.append(mask)

    all_masks = torch.cat(masks,0)
    preserve_num = int(all_masks.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(all_masks, preserve_num)
    threshold = preserve_weight[preserve_num-1]
    pr_cfg = []
    for mask in masks:
        pr_cfg.append(torch.sum(torch.lt(mask,threshold)).item()/mask.size(0))

    i = 0
    for n, m in model.named_modules():
        if hasattr(m, 'mask'):
            m.set_prune_rate(pr_cfg[i])
            i += 1

def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")
    
    # applying sparsity to the network
    if args.conv_type != "DenseConv":

        print(f"==> Setting prune rate of network to 0")

        def _sparsity(m):
            if hasattr(m, "set_prune_rate"):
                m.set_prune_rate(0)

        model.apply(_sparsity)
    

    return model

def adjust_learning_rate(optimizer, epoch):
    if epoch < 80:
        lr = args.lr
    elif epoch >= 80 and epoch < 120:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    mask_lr = lr * sigmoid(epoch-args.epochs/2, args.beta)
    optimizer.param_groups[0]['lr'] = mask_lr
    optimizer.param_groups[1]['lr'] = lr

    print('mask_lr: {}, para_lr: {}'.format(mask_lr, lr))


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            pass #print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            pass #print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        mask = [v for n, v in parameters if ("mask" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("mask" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": mask,
                    "weight_decay": 0
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer

def train(model, optimizer, trainLoader, args, epoch):
    model.train()
    losses = AverageMeter(':.4e')
    accurary = AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time

def validate(model, testLoader):
    global best_acc
    model.eval()

    losses = AverageMeter(':.4e')
    accurary = AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg

def set_gpu(args, model):
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model

import numpy as np

def sigmoid(x, beta):
  
    z = np.exp(-x*beta)
    sig = 1 / (1 + z)

    return sig

def main():
    start_epoch = 0
    best_acc = 0.0

    model = get_model(args)
    model = set_gpu(args, model)

    optimizer = get_optimizer(args, model)

    for epoch in range(start_epoch, args.epochs):
        prune_rate = args.prune_rate * sigmoid(epoch-args.epochs/2, args.beta)
        logger.info('Setting global pruning rate of the network to {:.3f}'.format(prune_rate))
        set_model(model, prune_rate)
        model = set_gpu(args, model)
        train(model, optimizer, loader.trainLoader, args, epoch)
        test_acc = validate(model, loader.testLoader)
        adjust_learning_rate(optimizer, epoch)

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)
        count = 0
        sum_sparse = 0.0
        for n, m in model.named_modules():
            if hasattr(m, "mask"):
                sparsity, total_params = m.getSparsity()
                logger.info("epoch{} {} sparsity {}% ".format(epoch, n, sparsity))
                sum_sparse += int(((100 - sparsity) / 100) * total_params)
                count += total_params
        total_sparsity = 100 - (100 * sum_sparse / count)
        logger.info("epoch {} sparsitytotal {}".format(epoch, total_sparsity))

    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

def resume(args, model, optimizer):
    if os.path.exists(args.job_dir+'/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")

        checkpoint = torch.load(args.job_dir+'/checkpoint/model_last.pt')

        start_epoch = checkpoint["epoch"]

        best_acc = checkpoint["best_acc"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")

        return start_epoch, best_acc

    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")



if __name__ == '__main__':
    main()