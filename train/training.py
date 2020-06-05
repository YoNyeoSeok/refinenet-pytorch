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


def arg_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--input-scale-factor', type=float, default=1.)
    parser.add_argument('--freeze-batch-norm', action='store_true')

    parser.add_argument('--clear-foggy-beta', type=str, default='clear', choices=['clear', 'beta_0.02', 'beta_0.01', 'beta_0.005'])
    parser.add_argument('--total-epoch', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--valid-batch-size', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--optimizer-lr', type=float, default=5e-5)

    parser.add_argument('--data-aug-hflip', action='store_true')
    parser.add_argument('--data-aug-hflip-p', type=float, default=0.5)
    parser.add_argument('--data-aug-crop', action='store_true')
    parser.add_argument('--data-aug-crop-size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--data-aug-crop-scale', type=float, nargs=2, default=[0.7, 1.3])
    parser.add_argument('--data-aug-crop-ratio', type=float, nargs=2, default=[1, 1])

    # parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-wandb', action='store_true')
    return parser

def load_train_valid_loader(args):
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

    if 'clear' == args.clear_foggy_beta:
        image_modes = ['clear', 'gtFine'] 
        image_types = [['_leftImg8bit.png'], ['semantic', 'color']]
    else:
        image_modes = ['foggyDBF', 'gtFine'] 
        image_types = [[args.clear_foggy_beta], ['semantic', 'color']]
    train_image_transforms = [aug_tensor_transform, [aug_semantic_transform, aug_image_transform]]
    valid_image_transforms = [tensor_transform, [semantic_transform, image_transform]]
    
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

def load_model(args):
    model_pretrained_dir = '/home/user/research/refinenet-pytorch/pretrained/Cityscapes'
    model = refinenet_resnet101(model_pretrained_dir)
    return model

# train_dl, valid_dl = load_train_valid_loader(args)
class InputOutputInterpolate(torch.nn.Module):
    def __init__(self, model, scale_factor):
        super(InputOutputInterpolate, self).__init__()
        self.model = model
        self.scale_factor = scale_factor

    def forward(self, x):
        size = x.shape[-2:]
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = self.model(x)
        return torch.nn.functional.interpolate(out, size=size, mode='bilinear', align_corners=False)

def load_training_model(args):
    training_model = InputOutputInterpolate(load_model(args), args.input_scale_factor)
    training_model.model.resnet.to(1)
    training_model.model.refinenets.to(0)
    training_model.model.clf.to(0)
    return training_model

class ModelCriteria(torch.nn.Module):
    def __init__(self, model, criteria):
        super(ModelCriteria, self).__init__()
        self.model = model
        self.criteria = criteria
    
    def forward(self, input, target):
        output = self.model(input)
        return self.criteria(output, target.to(output.device))

    def state_dict(self):
        return self.model.state_dict()

class ModelOptimizer(torch.nn.Module):
    def __init__(self, model_criteria, optimizer):
        super(ModelOptimizer, self).__init__()
        self.model_criteria = model_criteria
        self.optimizer = optimizer

    def step(self, input, target):
        self.optimizer.zero_grad()
        loss = self.model_criteria(input, target)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

def load_model_criteria_optimizer(args):
    model = load_training_model(args)

    CELoss = torch.nn.CrossEntropyLoss(ignore_index=255)
    # L1Loss = torch.nn.L1Loss()
    # L2Loss = torch.nn.MSELoss()
    model_criteria = ModelCriteria(model, CELoss)
    
    optimizer = torch.optim.__dict__[args.optimizer](
        model.parameters(),
        **{k.lstrip('optimizer_'): v for k, v in vars(args).items() if 'optimizer_' in k})
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    model_optimizer = ModelOptimizer(model_criteria, optimizer)

    return model, model_criteria, model_optimizer

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
            wandb.log({'Valid_Epoch_Loss': valid_epoch_loss}, step=self.train_batch_step)
        self.valid_epoch_step += 1

def train_model(model_optimizer, train_dl, wandb_log, args):
    train_loss = 0
    device = next(model_optimizer.parameters()).device
    model_optimizer.train()
    if args.freeze_batch_norm:
        for module in model_optimizer.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        
    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    # for _, ((b_clear_beta, ), (b_sparse, _)) in pbar:
    for _, ((b_clear_beta, ), (b_sparse, b_color)) in pbar:
        loss = model_optimizer.step(b_clear_beta.to(device), b_sparse)
        train_loss += loss

        pbar.set_description("Train Batch {:3d}".format(wandb_log.train_batch_step))
        pbar.set_postfix_str("Batch Loss={:.4f}".format(loss))
        wandb_log.train_batch_log(loss)

        train_dl.dataset.update_share_transform()
    train_loss /= len(train_dl)
    pbar.write("Train Epoch Loss={:.4f}".format(train_loss))
    return train_loss

def eval_model(model_criteria, valid_dl, wandb_log, args):
    eval_loss = 0
    device = next(model_criteria.parameters()).device
    model_criteria.eval()
    pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
    with torch.no_grad():
        for _, ((b_clear_beta, ), (b_sparse, _)) in pbar:
            loss = model_criteria(b_clear_beta.to(device), b_sparse).cpu().numpy()
            eval_loss += loss

            pbar.set_description("Valid Epoch {:3d}".format(wandb_log.valid_epoch_step))
    eval_loss /= len(valid_dl)
    pbar.write("Valid Epoch Loss={:.4f}".format(eval_loss))
    if wandb_log.use_wandb:
        torch.save(model_criteria.state_dict(), os.path.join(wandb.run.dir, 'state_dict.{:02d}.pth').format(wandb_log.valid_epoch_step))
    wandb_log.valid_epoch_log(eval_loss)
    return eval_loss

def main(parser, name, load_train_valid_loader, load_model_criteria_optimizer, train_model, eval_model):
    args = parser.parse_args()
    print(args)
    if args.use_wandb:
        wandb.init(project='refinenet-pytorch', name=name, config=args, dir='/home/user/research/refinenet-pytorch/train')

    train_dl, valid_dl = load_train_valid_loader(args)
    # train_dl.dataset.indices = train_dl.dataset.indices[:10]
    # valid_dl.dataset.indices = valid_dl.dataset.indices[:10]
    print('dataset loaded')
    model, model_criteria, model_optimizer = load_model_criteria_optimizer(args)
    print('model loaded')
    wandb_log = WandbLog(args.use_wandb)

    # eval_model(model_criteria, valid_dl, wandb_log, args)
    for epoch in range(args.total_epoch):
        train_model(model_optimizer, train_dl, wandb_log, args)
        eval_model(model_criteria, valid_dl, wandb_log, args)

if __name__ == '__main__':
    main(
        parser=arg_parser(),
        name='training',
        load_train_valid_loader=load_train_valid_loader,
        load_model_criteria_optimizer=load_model_criteria_optimizer,
        train_model=train_model,
        eval_model=eval_model)