import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from models.feature_regression import FeatureRegressionModel, FeatureRegressionModelCriteria

import wandb

from train import training

def arg_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--update-period', type=int, default=-1)
    parser.add_argument('--feature-regression-criteria', type=str, default='L1Loss', choices=['L1Loss', 'MSELoss'])
    parser.add_argument('--feature-regression-target-weight', type=float, nargs=2, default=[1, 1])
    parser.add_argument('--feature_layer', type=str, default='refinenets', choices=['refinenets', 'clf'])
    return parser

class FeatureRegressionInputOutputInterpolate(nn.Module):
    def __init__(self, model, scale_factor):
        super(FeatureRegressionInputOutputInterpolate, self).__init__()
        self.model = model
        self.scale_factor = scale_factor

    def forward(self, x):
        shape = x.shape[-2:]
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        out = self.model(x)
        return torch.nn.functional.interpolate(out[0], shape, mode='bilinear'), out[1], out[2]

def load_feature_regression_model(args):
    cuda_mapping = {
        'model': 'cuda:0',
        'target_model': 'cuda:1'}
    feature_regression_model = FeatureRegressionModel(
        model=training.load_model(args),
        target_model=training.load_model(args),
        feature_layer=args.feature_layer,
        cuda_mapping=cuda_mapping)
    feature_regression_model = FeatureRegressionInputOutputInterpolate(
        feature_regression_model,
        args.input_scale_factor)
    return feature_regression_model

def load_model_criteria_optimizer(args):
    feature_regression_model = load_feature_regression_model(args)

    CELoss = nn.modules.loss.CrossEntropyLoss()
    feature_criteria = nn.modules.loss.__dict__[args.feature_regression_criteria]()
    feature_regression_model_criteria = FeatureRegressionModelCriteria(
        feature_regression_model,
        feature_regression_criteria=feature_criteria,
        target_criteria=CELoss,
        feature_regression_target_weight=args.feature_regression_target_weight)
    
    optimizer = torch.optim.__dict__[args.optimizer](
        feature_regression_model.parameters(),
        **{k.lstrip('optimizer_'): v for k, v in vars(args).items() if 'optimizer_' in k})
    model_optimizer = training.ModelOptimizer(feature_regression_model_criteria, optimizer)

    return feature_regression_model, feature_regression_model_criteria, model_optimizer

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
        name='feature_regression_training',
        load_train_valid_loader=training.load_train_valid_loader,
        load_model_criteria_optimizer=load_model_criteria_optimizer,
        train_model=train_model,
        eval_model=eval_model)