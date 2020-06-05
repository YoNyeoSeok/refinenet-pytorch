import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse
import glob

import torch
import torch.nn as nn
import datasets as ds
from torchvision import transforms as trf
from models.refinenet_resnet import refinenet_resnet101
from utils.metrics import runningScore
from train import training
from eval import evaluating

import wandb
import yaml
import json


def main(parser, name, load_valid_test_loader, load_model, eval_model):
    args = parser.parse_args()
    print(args)
    if args.use_wandb:
        wandb.init(project='refinenet-pytorch', name=name, config=args, dir='/home/user/research/refinenet-pytorch/train')

    for wandb_path in sorted(glob.glob(os.path.join('/home/user/research/refinenet-pytorch/train/wandb', 'run*'))):
        if args.wandb_id == wandb_path[-8:]:
            break
    with open(os.path.join(wandb_path, 'wandb-metadata.json')) as f:
        metadata = json.load(f)
        assert metadata['name'] == 'joint-training' 
    with open(os.path.join(wandb_path, 'config.yaml')) as f:
        metadata = yaml.load(f, Loader=yaml.BaseLoader)
        args.input_scale_factor = float(metadata['input_scale_factor']['value'])

    valid_dl, test_dl = load_valid_test_loader(args)
    # valid_dl.dataset.indices = valid_dl.dataset.indices[:4]
    # test_dl.dataset.images = test_dl.dataset.images[:3]
    print('dataset loaded')
    model = load_model(args)
    print('model loaded')
    wandb_log = evaluating.WandbLog(args.use_wandb)

    for state_dict_path in sorted(glob.glob(os.path.join(wandb_path, 'state_dict.*.pth'))):
        epoch = int(state_dict_path[-len('state_dict.00.pth'):].lstrip('state_dict.').rstrip('.pth'))
        wandb_log.running_metrics_epoch_step = epoch
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        eval_model(model, valid_dl, test_dl, wandb_log, args)

if __name__ == '__main__':
    main(
        parser=evaluating.arg_parser(),
        name='joint-evaluating',
        load_valid_test_loader=evaluating.load_valid_test_loader,
        load_model=training.load_training_model,
        eval_model=evaluating.eval_model)