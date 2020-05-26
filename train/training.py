import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse

import torch
import torch.nn as nn
import datasets as ds
from torchvision import transforms as trf
from models.refinenet_resnet import refinenet_resnet101
from utils.metrics import runningScore

import wandb

def arg_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--foggy_beta', type=str, default='beta_0.02', choices=['beta_0.02', 'beta_0.01', 'beta_0.005'])
    parser.add_argument('--input_scale_factor', type=float, default=7/16)
    parser.add_argument('--total-epoch', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--valid_batch_size', type=int, default=3)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--optimizer_lr', type=float, default=5e-5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-wandb', action='store_true')
    return parser

def load_train_valid_loader(args):
    cityscape_dataset_dir = '/home/user/data/Cityscapes'
    classes = ds.Cityscapes.classes
    id2label = {cls.id:cls for cls in classes}

    def semantic2sparse(semantic):
        sparse = np.vectorize(lambda x: id2label[x].train_id)(np.array(semantic))
        sparse = np.vectorize(lambda x: 19 if x == 255 else x)(sparse)
        # pylint: disable=E1101
        sparse = torch.from_numpy(sparse)
        # pylint: enable=E1101
        return sparse

    train_transform = trf.Compose([
        # trf.RandomHorizontalFlip(),
        trf.ToTensor(),
        trf.Lambda(lambda x: x*255-128),
    ])
    val_transform = trf.Compose([
        trf.ToTensor(),
    ])
    semantic_transform = trf.Compose([
        trf.Lambda(semantic2sparse),
    ])
    image_transform = trf.Compose([
        trf.Lambda(np.array)
    ])

    train_ds = ds.RefinedFoggyCityscapes(
        cityscape_dataset_dir,
        split='train',
        image_modes=['foggyDBF', 'gtFine'], 
        image_types=[[args.foggy_beta], ['semantic', 'color']],
        image_transforms=[train_transform, [semantic_transform, image_transform]],
        refined_filenames='foggy_trainval_refined_filenames.txt')
    valid_ds = ds.RefinedFoggyCityscapes(
        cityscape_dataset_dir,
        split='val',
        image_modes=['foggyDBF', 'gtFine'], 
        image_types=[[args.foggy_beta], ['semantic', 'color']],
        image_transforms=[val_transform, [semantic_transform, image_transform]],
        refined_filenames='foggy_trainval_refined_filenames.txt')

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=args.valid_batch_size, shuffle=False)
    return train_dl, valid_dl

# train_dl, valid_dl = load_train_valid_loader(args)
def load_model(args):
    model_pretrained_dir = '/home/user/research/refinenet-pytorch/pretrained/Cityscapes'
    model = refinenet_resnet101(model_pretrained_dir)
    class InputOutputInterpolate(torch.nn.Module):
        def __init__(self, model, scale_factor):
            super(InputOutputInterpolate, self).__init__()
            self.model = model
            self.scale_factor = scale_factor

        def forward(self, x):
            shape = x.shape[-2:]
            x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
            out = self.model(x)
            return torch.nn.functional.interpolate(out, shape, mode='bilinear')
    model = InputOutputInterpolate(model, args.input_scale_factor)
    return model

def load_model_criteria_optimizer(args):
    model = load_model(args)
    CELoss = torch.nn.CrossEntropyLoss()
    # L1Loss = torch.nn.L1Loss()
    # L2Loss = torch.nn.MSELoss()
    model_criteria = ModelCriteria(model, CELoss)
    
    optimizer = torch.optim.__dict__[args.optimizer](
        model.parameters(),
        **{k.lstrip('optimizer_'): v for k, v in vars(args).items() if 'optimizer_' in k})
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    model_optimizer = ModelOptimizer(model_criteria, optimizer)

    return model, model_criteria, model_optimizer

class ModelCriteria(torch.nn.Module):
    def __init__(self, model, criteria):
        super(ModelCriteria, self).__init__()
        self.model = model
        self.criteria = criteria
    
    def forward(self, input, target):
        return self.criteria(self.model(input), target)


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

    # def step_loop(self, epoch, train_dl):
    #     self.step += 1
    #     self.model.train()
    #     pbar = tqdm(enumerate(train_dl), total=len(train_dl))
    #     for b, ((b_beta, ), (b_sparse, _)) in pbar:
    #         loss = self.step(b_beta, b_sparse)
    #         self.batch_info = {'batch': batch,
    #                            'loss': loss.detach().cpu().numpy(),
    #                            'step': self.step}
            
    #         self.batch_end_func()

    #         train_loss += loss.detach().cpu().numpy()
    #     train_loss /= len(train_dl)
    #     return train_loss

    # def on_batch_end(self):
    #     if args.use_wandb:
    #         wandb.log({'Train_Batch_Loss': loss}, step=self.step)
    #     pbar.set_description("Batch {:3d}".format(b))
    #     pbar.set_postfix(Loss=loss.detach().cpu().numpy())

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
    model_optimizer.to(args.gpu)
    model_optimizer.train()
    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    for _, ((b_beta, ), (b_sparse, _)) in pbar:
        loss = model_optimizer.step(b_beta.to(args.gpu), b_sparse.to(args.gpu))
        train_loss += loss

        pbar.set_description("Train Batch {:3d}".format(wandb_log.train_batch_step))
        pbar.set_postfix_str("Batch Loss={:.4f}".format(loss))
        wandb_log.train_batch_log(loss)
    train_loss /= len(train_dl)
    pbar.write("Train Epoch Loss={:.4f}".format(train_loss))
    return train_loss

def eval_model(model_criteria, valid_dl, wandb_log, args):
    eval_loss = 0
    model_criteria.to(args.gpu)
    model_criteria.eval()
    pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
    with torch.no_grad():
        for _, ((b_beta, ), (b_sparse, _)) in pbar:
            loss = model_criteria(b_beta.to(args.gpu), b_sparse.to(args.gpu)).cpu().numpy()
            eval_loss += loss            

            pbar.set_description("Valid Epoch {:3d}".format(wandb_log.valid_epoch_step))
    eval_loss /= len(valid_dl)
    pbar.write("Valid Epoch Loss={:.4f}".format(eval_loss))
    if wandb_log.use_wandb:
        torch.save(model_criteria.model.state_dict(), os.path.join(wandb.run.dir, 'state_dict.{:02d}.pth').format(wandb_log.valid_epoch_step))
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

    eval_model(model_criteria, valid_dl, wandb_log, args)
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