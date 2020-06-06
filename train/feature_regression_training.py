import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse
from collections import OrderedDict, Counter

import torch
import torch.nn as nn

import wandb

from train import training, joint_training
from torchvision.models._utils import IntermediateLayerGetter

def arg_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--update-period', type=int, default=-1)
    parser.add_argument('--feature-regression-criteria', type=str, default='L1Loss', choices=['L1Loss', 'MSELoss'])
    parser.add_argument('--feature-regression-target-weights', type=float, nargs=2, default=[1, 1])
    parser.add_argument('--feature_layer', type=str, default='refinenets', choices=['refinenets', 'clf', 'clf_interp'])
    return parser


class InputOutputInterpolateIntermediateLayerGetter(IntermediateLayerGetter):
    def __init__(self, model, return_layers, interp_layer, scale_factor):
        super(InputOutputInterpolateIntermediateLayerGetter, self).__init__(
            model, return_layers,
        )
        self.scale_factor = scale_factor
        self.interp_layer = interp_layer

    def forward(self, x):
        size = x.shape[-2:]
        x = torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        out = super(InputOutputInterpolateIntermediateLayerGetter, self).forward(x)
        out['interp'] = torch.nn.functional.interpolate(
            out[self.interp_layer], size=size, mode='bilinear', align_corners=False)
        return out

class FeatureRegressionModel(nn.ModuleDict):
    def __init__(self, model, target_model, feature_layer, predict_layer):
        super(FeatureRegressionModel, self).__init__({
            'model': model,
            'target_model': target_model})
        self.feature_layer = feature_layer
        self.predict_layer = predict_layer

    def forward(self, input):
        out = OrderedDict()

        device = next(self.model.parameters()).device
        x = self.model(input.to(device))
        out['model_feature'] = x[self.feature_layer]
        out['model_predict'] = x[self.predict_layer]

        device = next(self.target_model.parameters()).device
        x = self.target_model(input.to(device))
        out['target_model_feature'] = x[self.feature_layer]
        out['target_model_predict'] = x[self.predict_layer]

        return out

    def train(self, mode=True):
        self.model.train(mode)

def load_feature_interp_model(args):
    model = training.load_model(args)
    model = InputOutputInterpolateIntermediateLayerGetter(
        model,
        return_layers={'refinenets': 'refinenets', 'clf': 'clf'},
        interp_layer='clf',
        scale_factor=args.input_scale_factor)
    return model

def load_feature_regression_model(args):
    model = load_feature_interp_model(args)
    target_model = load_feature_interp_model(args)

    feature_regression_model = FeatureRegressionModel(
        model=model,
        target_model=target_model,
        feature_layer=args.feature_layer if args.feature_layer != 'clf_interp' else 'interp',
        predict_layer='interp')
    feature_regression_model.model.to('cuda:1')
    feature_regression_model.target_model.to('cuda:0')
    return feature_regression_model

class FeatureRegressionModelCriteria(nn.Module):
    def __init__(
        self,
        feature_regression_model,
        feature_regression_criteria,
        target_criteria,
        feature_regression_target_weights,
        ):

        super(FeatureRegressionModelCriteria, self).__init__()
        self.feature_regression_model = feature_regression_model
        self.feature_regression_criteria = feature_regression_criteria
        self.target_criteria = target_criteria
        # pylint: disable=E1101
        self.weights = torch.FloatTensor(feature_regression_target_weights)
        # pylint: enable=E1101

    def forward(self, input, target=None):
        out = self.feature_regression_model(input)

        feature = out['model_feature']
        logit = out['model_predict']
        target_feature = out['target_model_feature']
        # out['target_model_predict']

        # pylint: disable=E1101
        feature_regression_loss = self.feature_regression_criteria(feature, target_feature.to(feature.device),)
        target_loss = (
            torch.FloatTensor(0).to(logit.device)
            if target is None else self.target_criteria(logit, target.to(logit.device))
        )
        losses = torch.stack([
            feature_regression_loss,
            target_loss
        ])
        # pylint: enable=E1101
        return OrderedDict([
            ('feature_regression', losses[0]),
            ('target', losses[1]),
            ('total', losses @ self.weights.to(losses.device))])

    def state_dict(self):
        return self.feature_regression_model.state_dict()

class FeatureRegressionModelOptimizer(nn.Module):
    def __init__(self, feature_regression_model_criteria, optimizer):
        super(FeatureRegressionModelOptimizer, self).__init__()
        self.feature_regression_model_criteria = feature_regression_model_criteria
        self.optimizer = optimizer

    def step(self, input, target):
        self.optimizer.zero_grad()
        loss = self.feature_regression_model_criteria(input, target)
        loss['total'].backward()
        self.optimizer.step()

        return loss

def load_feature_regression_model_criteria_optimizer(args):
    feature_regression_model = load_feature_regression_model(args)

    CELoss = torch.nn.CrossEntropyLoss(ignore_index=255)
    feature_criteria = nn.modules.loss.__dict__[args.feature_regression_criteria]()
    feature_regression_model_criteria = FeatureRegressionModelCriteria(
        feature_regression_model,
        feature_regression_criteria=feature_criteria,
        target_criteria=CELoss,
        feature_regression_target_weights=args.feature_regression_target_weights)
    
    optimizer = torch.optim.__dict__[args.optimizer](
        feature_regression_model.parameters(),
        **{k.lstrip('optimizer_'): v for k, v in vars(args).items() if 'optimizer_' in k})
    model_optimizer = FeatureRegressionModelOptimizer(feature_regression_model_criteria, optimizer)

    return feature_regression_model, feature_regression_model_criteria, model_optimizer

WandbLog = joint_training.WandbLog

def train_model(model_optimizer, train_dl, wandb_log, args):
    train_loss = Counter()
    device = next(model_optimizer.parameters()).device
    model_optimizer.train()
    if args.freeze_batch_norm:
        for module in model_optimizer.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    for _, ((b_clear_beta, ), (b_sparse, b_color)) in pbar:
        loss = model_optimizer.step(b_clear_beta.to(device), b_sparse)
        train_loss += loss

        pbar.set_description("Train Batch {:3d}".format(wandb_log.train_batch_step))
        pbar.set_postfix_str("Batch Loss={:.4f}".format(loss['total'].detach().cpu().numpy()))
        wandb_log.train_batch_log(loss['total'].detach().cpu().numpy())

        train_dl.dataset.update_share_transform()
    train_loss = Counter(OrderedDict([(k, v/len(train_dl)) for k, v in train_loss.items()]))
    pbar.write("Train Epoch Loss={:.4f}".format(train_loss['total'].detach().cpu().numpy()))
    return train_loss


def eval_model(model_criteria, valid_dl, wandb_log, args):
    eval_loss = Counter()
    device = next(model_criteria.parameters()).device
    model_criteria.eval()
    
    pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
    with torch.no_grad():
        for _, ((b_clear_beta, ), (b_sparse, _)) in pbar:
            loss = model_criteria(b_clear_beta.to(device), b_sparse)
            eval_loss += Counter(loss)

            pbar.set_description("Valid Epoch {:3d}".format(wandb_log.valid_epoch_step))
    eval_loss = Counter(OrderedDict([(k, v/len(valid_dl)) for k, v in eval_loss.items()]))
    pbar.write("Valid Epoch Loss={:.4f}".format(eval_loss['total'].cpu().numpy()))
    if wandb_log.use_wandb:
        state_dict_name = 'state_dict.{:02d}.pth'.format(wandb_log.valid_epoch_step)
        torch.save(model_criteria.state_dict(), os.path.join(wandb.run.dir, state_dict_name))
        wandb.save(state_dict_name)
    wandb_log.valid_epoch_log(OrderedDict([(k, v.cpu().numpy()) for k, v in eval_loss.items()]))
    return eval_loss

if __name__ == '__main__':
    training.main(
        parser=arg_parser(training.arg_parser()),
        name='feature-regression-training',
        load_train_valid_loader=training.load_train_valid_loader,
        load_model_criteria_optimizer=load_feature_regression_model_criteria_optimizer,
        train_model=train_model,
        eval_model=eval_model)