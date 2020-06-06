import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse
import math
import random
from PIL import Image

import torch
import torch.nn as nn
import datasets as ds
from torchvision import transforms as trf
from models.refinenet_resnet import refinenet_resnet101
from utils.metrics import runningScore
from vision.transforms import RandomHorizontalFlip, RandomResizedCrop

import wandb

from train import training
from collections import OrderedDict, Counter

def arg_parser(parser=argparse.ArgumentParser()):
    # parser.add_argument('--input-scale-factor', type=float, default=1.)
    # parser.add_argument('--freeze-batch-norm', action='store_true')
    
    parser.add_argument('--clear-foggy-beta', type=str, default=['clear', 'beta_0.005', 'beta_0.01', 'beta_0.02'], nargs='+', choices=['clear', 'beta_0.02', 'beta_0.01', 'beta_0.005'])
    parser.add_argument('--clear-foggy-weights', type=float, default=[1., 1., 1., 1.], nargs='+')

    # parser.add_argument('--total-epoch', type=int, default=12)
    # parser.add_argument('--batch-size', type=int, default=1)
    # parser.add_argument('--valid-batch-size', type=int, default=3)
    # parser.add_argument('--optimizer', type=str, default='SGD')
    # parser.add_argument('--optimizer-lr', type=float, default=5e-5)

    # parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--use-wandb', action='store_true')
    return parser

def load_train_valid_joint_loader(args):
    cityscape_dataset_dir = '/home/user/data/Cityscapes'
    classes = ds.Cityscapes.classes
    id2label = {cls.id:cls for cls in classes}

    def semantic2sparse(semantic):
        sparse = np.vectorize(lambda x: id2label[x].train_id)(np.array(semantic))
        # pylint: disable=E1101
        sparse = torch.from_numpy(sparse)
        # pylint: enable=E1101
        return sparse

    hflip = [RandomHorizontalFlip(args.data_aug_hflip_p)] if args.data_aug_hflip else []
    resized_crop = ([
        RandomResizedCrop(args.data_aug_crop_size, args.data_aug_crop_scale, args.data_aug_crop_ratio)] 
        if args.data_aug_crop else [])

    aug_transform = trf.Compose(hflip + resized_crop)

    tensor_transform = trf.Compose([
            trf.ToTensor(),
            trf.Lambda(lambda x: x*255-128),
    ])
    semantic_transform = trf.Compose([
        trf.Lambda(semantic2sparse),
        ])
    image_transform = trf.Compose([
        trf.Lambda(np.array)
        ])

    aug_tensor_transform = trf.Compose(
        aug_transform.transforms + tensor_transform.transforms)
    aug_semantic_transform = trf.Compose(
        aug_transform.transforms + semantic_transform.transforms)
    aug_image_transform = trf.Compose(
        aug_transform.transforms + image_transform.transforms)

    image_modes = []
    image_types = []
    train_image_transforms = []
    valid_image_transforms = []

    for clear_foggy_beta in args.clear_foggy_beta:
        if clear_foggy_beta == 'clear':
            image_modes += ['clear']
            image_types += [['_leftImg8bit.png']]
        else:
            image_modes += ['foggyDBF']
            image_types += [[clear_foggy_beta]]
        train_image_transforms += [aug_tensor_transform]
        valid_image_transforms += [tensor_transform]

    image_modes += ['gtFine']
    image_types += [['semantic', 'color']]
    train_image_transforms += [[aug_semantic_transform, aug_image_transform]]
    valid_image_transforms += [[semantic_transform, image_transform]]

    train_ds = ds.RefinedFoggyCityscapes(
        cityscape_dataset_dir,
        split='train',
        image_modes=image_modes, 
        image_types=image_types,
        image_transforms=train_image_transforms,
        refined_filenames='foggy_trainval_refined_filenames.txt')
    train_ds.share_transform = aug_transform
    train_ds.update_share_transform = lambda : [transform.update() for transform in aug_transform.transforms]

    valid_ds = ds.RefinedFoggyCityscapes(
        cityscape_dataset_dir,
        split='val',
        image_modes=image_modes, 
        image_types=image_types,
        image_transforms=valid_image_transforms,
        refined_filenames='foggy_trainval_refined_filenames.txt')

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=args.valid_batch_size, shuffle=False)

    return train_dl, valid_dl

load_training_model = training.load_training_model

class ModelJointCriteria(torch.nn.Module):
    def __init__(self, model, criteria, weights):
        super(ModelJointCriteria, self).__init__()
        self.model = model
        self.criteria = criteria
        self.weights = torch.Tensor(weights)
        nn.CrossEntropyLoss()
        
    def forward(self, names, inputs, target, batch=True):
        b_target = target.repeat_interleave(len(inputs), dim=0)
        # pylint: disable=E1101
        b_input = torch.cat(inputs, dim=0)
        # pylint: enable=E1101

        if batch:
            b_output = self.model(b_input)
            b_loss = self.criteria(b_output, b_target.to(b_output.device))
            # pylint: disable=E1101
            losses = torch.stack([loss.mean() for loss in b_loss.split([len(input) for input in inputs], dim=0)])
            # pylint: enable=E1101
        else:
            outputs = [self.model(b_input) for b_input in inputs]
            losses = [self.criteria(b_output, target.to(b_output.device)) for b_output in outputs]
            # pylint: disable=E1101
            losses = torch.stack([loss.mean() for loss in losses])
            # pylint: enable=E1101
        return OrderedDict(list(zip(names, losses)) + [('total', losses @ self.weights.to(losses[0].device))])

    def state_dict(self):
        return self.model.state_dict()

class ModelJointOptimizer(torch.nn.Module):
    def __init__(self, model_joint_criteria, optimizer):
        super(ModelJointOptimizer, self).__init__()
        self.model_joint_criteria = model_joint_criteria
        self.optimizer = optimizer

    def step(self, names, inputs, target):
        self.optimizer.zero_grad()
        loss = self.model_joint_criteria(names, inputs, target)
        loss['total'].backward()
        self.optimizer.step()

        return loss

def load_model_joint_criteria_optimizer(args):
    model = load_training_model(args)

    CELoss = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    # L1Loss = torch.nn.L1Loss()
    # L2Loss = torch.nn.MSELoss()
    model_joint_criteria = ModelJointCriteria(model, CELoss, args.clear_foggy_weights)
    
    optimizer = torch.optim.__dict__[args.optimizer](
        model.parameters(),
        **{k.lstrip('optimizer_'): v for k, v in vars(args).items() if 'optimizer_' in k})
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    model_joint_optimizer = ModelJointOptimizer(model_joint_criteria, optimizer)

    return model, model_joint_criteria, model_joint_optimizer

class WandbLog():
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb
        self.train_batch_step = 0
        self.valid_epoch_step = 0
    def train_batch_log(self, train_batch_loss):
        self.train_batch_step += 1
        if self.use_wandb:
            wandb.log({'Train_Batch_Loss': train_batch_loss}, step=self.train_batch_step)
    def valid_epoch_log(self, valid_epoch_loss):
        if self.use_wandb:
            wandb.log({'Valid_Epoch_Loss_({})'.format(k):v for k, v in valid_epoch_loss.items()}, step=self.train_batch_step)
        self.valid_epoch_step += 1

def train_model(model_optimizer, train_dl, wandb_log, args):
    train_loss = Counter()
    device = next(model_optimizer.parameters()).device
    model_optimizer.train()
    if args.freeze_batch_norm:
        for module in model_optimizer.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    for batch, b_clear_beta_semantic_color in pbar:
        b_clear_beta = b_clear_beta_semantic_color[:-1]
        b_sparse, b_color = b_clear_beta_semantic_color[-1]

        names = args.clear_foggy_beta
        inputs = sum(b_clear_beta, [])

        loss = model_optimizer.step(names, [input.to(device) for input in inputs], b_sparse)
        train_loss += Counter(loss)

        pbar.set_description("Train Batch {:3d}".format(wandb_log.train_batch_step))
        pbar.set_postfix_str("Batch Loss={:.4f}".format(loss['total'].detach().cpu().numpy()))
        wandb_log.train_batch_log(loss['total'].detach().cpu().numpy())
    train_loss = Counter(OrderedDict([(k, v/len(train_dl)) for k, v in train_loss.items()]))
    pbar.write("Train Epoch Loss={:.4f}".format(train_loss['total'].detach().cpu().numpy()))
    return train_loss

def eval_model(model_joint_criteria, valid_dl, wandb_log, args):
    eval_loss = Counter()
    device = next(model_joint_criteria.parameters()).device
    model_joint_criteria.eval()

    pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
    with torch.no_grad():
        for batch, b_clear_beta_semantic_color in pbar:
            b_clear_beta = b_clear_beta_semantic_color[:-1]
            b_sparse, b_color = b_clear_beta_semantic_color[-1]

            names = args.clear_foggy_beta
            inputs = sum(b_clear_beta, [])
            
            loss = model_joint_criteria(names, [input.to(device) for input in inputs], b_sparse, batch=False)
            eval_loss += Counter(loss)

            pbar.set_description("Valid Epoch {:3d}".format(wandb_log.valid_epoch_step))
    eval_loss = Counter(OrderedDict([(k, v/len(valid_dl)) for k, v in eval_loss.items()]))
    pbar.write("Valid Epoch Loss={:.4f}".format(eval_loss['total'].cpu().numpy()))
    if wandb_log.use_wandb:
        state_dict_name = 'state_dict.{:02d}.pth'.format(wandb_log.valid_epoch_step)
        torch.save(model_joint_criteria.state_dict(), os.path.join(wandb.run.dir, state_dict_name))
        wandb.save(state_dict_name)
    wandb_log.valid_epoch_log(OrderedDict([(k, v.cpu().numpy()) for k, v in eval_loss.items()]))
    return eval_loss

def main(parser, name, load_train_valid_loader, load_model_criteria_optimizer, train_model, eval_model):
    args = parser.parse_args()
    print(args)
    if args.use_wandb:
        wandb.init(project='refinenet-pytorch', name=name, config=args, dir='/home/user/research/refinenet-pytorch/train')

    train_dl, valid_dl = load_train_valid_loader(args)
    # train_dl.dataset.indices = train_dl.dataset.indices[:2]
    # valid_dl.dataset.indices = valid_dl.dataset.indices[:4]
    print('dataset loaded')
    model, model_criteria, model_optimizer = load_model_criteria_optimizer(args)
    print('model loaded')
    wandb_log = WandbLog(args.use_wandb)

    eval_model(model_criteria, valid_dl, wandb_log, args)
    for epoch in range(args.total_epoch):
        train_model(model_optimizer, train_dl, wandb_log, args)
        eval_model(model_criteria, valid_dl, wandb_log, args)

if __name__ == '__main__':
    main(
        parser=arg_parser(training.arg_parser(argparse.ArgumentParser(conflict_handler='resolve'))),
        name='joint-training',
        load_train_valid_loader=load_train_valid_joint_loader,
        load_model_criteria_optimizer=load_model_joint_criteria_optimizer,
        train_model=train_model,
        eval_model=eval_model)