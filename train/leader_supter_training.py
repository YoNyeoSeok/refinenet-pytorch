import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from models.leader_supter import LeaderSupterModel, LeaderSupterModelCriteria, CudaChildrenLayer

import wandb

from train import training

def arg_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--update-period', type=int, default=-1)
    parser.add_argument('--supter-criteria', type=str, default='L1Loss', choices=['L1Loss', 'MSELoss'])
    parser.add_argument('--supter-target-weight', type=float, nargs=2, default=[1, 1])
    return parser

def load_leader_supter_model(args):
    cuda_mapping = {
        'leader_model': 'cuda:0',
        'supter_model': 'cuda:1'}
    leader_supter_model = LeaderSupterModel(
        training.load_training_model(args),
        training.load_training_model(args),
        cuda_mapping=cuda_mapping)
    return leader_supter_model


def load_model_criteria_optimizer(args):
    leader_supter_model = load_leader_supter_model(args)

    CELoss = nn.modules.loss.CrossEntropyLoss()
    supter_criteria = nn.modules.loss.__dict__[args.supter_criteria]()
    # L1Loss = nn.L1Loss()
    # L2Loss = nn.MSELoss()
    leader_supter_model_criteria = LeaderSupterModelCriteria(
        leader_supter_model,
        supter_criteria=supter_criteria,
        target_criteria=CELoss,
        supter_target_weight=args.supter_target_weight)
    
    optimizer = torch.optim.__dict__[args.optimizer](
        leader_supter_model.parameters(),
        **{k.lstrip('optimizer_'): v for k, v in vars(args).items() if 'optimizer_' in k})
    model_optimizer = training.ModelOptimizer(leader_supter_model_criteria, optimizer)

    return leader_supter_model, leader_supter_model_criteria, model_optimizer

def train_model(model_optimizer, train_dl, wandb_log, args):
    train_loss = 0
    # model_optimizer.to(args.gpu)
    model_optimizer.train()
    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    for _, ((b_beta, ), (b_sparse, _)) in pbar:
        loss = model_optimizer.step(b_beta.to(args.gpu), b_sparse)
        train_loss += loss

        pbar.set_description("Train Batch {:3d}".format(wandb_log.train_batch_step))
        pbar.set_postfix_str("Batch Loss={:.4f}".format(loss))
        wandb_log.train_batch_log(loss)
    train_loss /= len(train_dl)
    pbar.write("Train Epoch Loss={:.4f}".format(train_loss))
    return train_loss

def eval_model(model_criteria, valid_dl, wandb_log, args):
    eval_loss = 0
    # model_criteria.to(args.gpu)
    model_criteria.eval()
    pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
    with torch.no_grad():
        for _, ((b_beta, ), (b_sparse, _)) in pbar:
            loss = model_criteria(b_beta.to(args.gpu), b_sparse).cpu().numpy()
            eval_loss += loss            

            pbar.set_description("Valid Epoch {:3d}".format(wandb_log.valid_epoch_step))
    eval_loss /= len(valid_dl)
    pbar.write("Valid Epoch Loss={:.4f}".format(eval_loss))
    if wandb_log.use_wandb:
        torch.save(model_criteria.state_dict(), os.path.join(wandb.run.dir, 'state_dict.{:02d}.pth').format(wandb_log.valid_epoch_step))
    wandb_log.valid_epoch_log(eval_loss)
    return eval_loss

if __name__ == '__main__':
    training.main(
        parser=arg_parser(training.arg_parser()),
        name='leader_supter_training',
        load_train_valid_loader=training.load_train_valid_loader,
        load_model_criteria_optimizer=load_model_criteria_optimizer,
        train_model=train_model,
        eval_model=eval_model)