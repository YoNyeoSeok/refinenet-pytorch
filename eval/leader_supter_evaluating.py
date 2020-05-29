import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse
import glob

import torch
import torch.nn as nn
from models.leader_supter import LeaderSupterModel
import datasets as ds
from torchvision import transforms as trf
from models.refinenet_resnet import refinenet_resnet101
from utils.metrics import runningScore
from train import leader_supter_training

import wandb
import yaml
import json
## 1th8v62u beta_0.02
# 1wmq3g1d beta_0.01
# 3dlxt8tg beta_0.005
def arg_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--wandb_id', type=str, default='3dlxt8tg')
    parser.add_argument('--valid_batch_size', type=int, default=3)
    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--use-wandb', action='store_true')
    return parser

def load_valid_test_loader(args):
    cityscape_dataset_dir = '/home/user/data/Cityscapes'
    classes = ds.Cityscapes.classes
    id2label = {cls.id:cls for cls in classes}

    zurich_dataset_dir = '/home/user/data/Foggy_Zurich'

    def semantic2sparse(semantic):
        sparse = np.vectorize(lambda x: id2label[x].train_id)(np.array(semantic))
        sparse = np.vectorize(lambda x: 19 if x == 255 else x)(sparse)
        # pylint: disable=E1101
        sparse = torch.from_numpy(sparse)
        # pylint: enable=E1101
        return sparse

    valid_transform = trf.Compose([
        trf.ToTensor(),
        trf.Lambda(lambda x: x*255-128),
    ])
    test_transform = trf.Compose([
        trf.ToTensor(),
        trf.Lambda(lambda x: x*255-128),
    ])
    semantic_transform = trf.Compose([
        trf.Lambda(semantic2sparse),
    ])
    image_transform = trf.Compose([
        trf.Lambda(np.array)
    ])

    valid_ds = ds.RefinedFoggyCityscapes(
        cityscape_dataset_dir,
        split='val',
        image_modes=['clear', 'foggyDBF', 'gtFine'], 
        image_types=[[None], ['beta_0.005', 'beta_0.01', 'beta_0.02'], ['semantic', 'color']],
        image_transforms=[valid_transform, valid_transform, [semantic_transform, image_transform]],
        refined_filenames='foggy_trainval_refined_filenames.txt')
    test_ds = ds.Zurich(
        zurich_dataset_dir,
        split='testv2',
        image_modes=['RGB', 'semantic', 'color'],
        image_transforms=[test_transform, semantic_transform, image_transform],
    )

    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=args.valid_batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)
    return valid_dl, test_dl

# train_dl, valid_dl = load_train_valid_loader(args)
class WandbLog():
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb
        self.running_metrics_epoch_step = 0
    def running_metrics_epoch_log(self, name, running_metrics):
        metrics, per_class_IoU = running_metrics.get_scores()
        # print(metrics)
        # print(per_class_IoU)
        if self.use_wandb:
            wandb.log(
                {' '.join([metric, name, ]): value for metric, value in metrics.items()},
                step=self.running_metrics_epoch_step,)
            wandb.log(
                {' '.join([str(class_IoU), name, ]): value for class_IoU, value in per_class_IoU.items()},
                step=self.running_metrics_epoch_step,)

def eval_model(model, valid_dl, test_dl, wandb_log, args):
    model.to(args.gpu)
    model.eval()
    eval_running_metrics = [runningScore(20) for i in range(5)]
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
        for _, ((b_clear, ), (b_beta_005, b_beta_01, b_beta_02, ), (b_sparse, _, )) in pbar:
            for running_metrics, b_input in zip(eval_running_metrics[:4], [b_clear, b_beta_005, b_beta_01, b_beta_02, ]):
                b_sparse_pred = model(b_input.to(args.gpu)).argmax(1).cpu()
                running_metrics.update(b_sparse.numpy(), b_sparse_pred.numpy(), )
            pbar.set_description("Valid Epoch {:3d}".format(wandb_log.running_metrics_epoch_step))
        if wandb_log.use_wandb:
            for name, running_metrics in zip(['clear', 'beta_0.005', 'beta_0.01', 'beta_0.02', ], eval_running_metrics[:4]):
                wandb_log.running_metrics_epoch_log(name, running_metrics)
                
        pbar = tqdm.tqdm(enumerate(test_dl), total=len(test_dl))
        for _, (b_input, b_sparse, _, ) in pbar:
            b_sparse_pred = model(b_input.to(args.gpu)).argmax(1).cpu()
            eval_running_metrics[-1].update(b_sparse.numpy(), b_sparse_pred.numpy(), )
        if wandb_log.use_wandb:
            wandb_log.running_metrics_epoch_log('testv2', eval_running_metrics[-1])
            pbar.set_description("Test Epoch {:3d}".format(wandb_log.running_metrics_epoch_step))

    for name, running_metrics in zip(['clear', 'beta_0.005', 'beta_0.01', 'beta_0.02', 'testv2'], eval_running_metrics):
        metrics, per_class_IoU = running_metrics.get_scores()
        pbar.write("{} Evaluation Metrics={}".format(name, metrics))
        pbar.write("{} Evaluation per_class_IoU={}".format(name, per_class_IoU))
    return eval_running_metrics

def main(parser, name, load_valid_test_loader, load_model, eval_model):
    args = parser.parse_args()
    print(args)
    if args.use_wandb:
        wandb.init(project='refinenet-pytorch', name=name, config=args, dir='/home/user/research/refinenet-pytorch/train')

    for wandb_path in sorted(glob.glob(os.path.join('/home/user/research/refinenet-pytorch/train/wandb', 'run*'))):
        if args.wandb_id == wandb_path[-8:]:
            break
    assert args.wandb_id == wandb_path[-8:], "Couldn't fild wandb id {}".format(args.wandb_id)
    with open(os.path.join(wandb_path, 'wandb-metadata.json')) as f:
        metadata = json.load(f)
        assert metadata['name'] == 'leader_supter_training', metadata['name']
    with open(os.path.join(wandb_path, 'config.yaml')) as f:
        metadata = yaml.load(f, Loader=yaml.BaseLoader)
        args.input_scale_factor = float(metadata['input_scale_factor']['value'])

    valid_dl, test_dl = load_valid_test_loader(args)
    valid_dl.dataset.indices = valid_dl.dataset.indices[:18]
    # test_dl.dataset.images = test_dl.dataset.images[:3]
    print('dataset loaded')
    model = load_model(args).cpu()
    print('model loaded')
    wandb_log = WandbLog(args.use_wandb)

    for state_dict_path in sorted(glob.glob(os.path.join(wandb_path, 'state_dict.*.pth'))):
        epoch = int(state_dict_path[-len('state_dict.00.pth'):].lstrip('state_dict.').rstrip('.pth'))
        wandb_log.running_metrics_epoch_step = epoch
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        eval_model(model['leader_model'], valid_dl, test_dl, wandb_log, args)

if __name__ == '__main__':
    main(
        parser=arg_parser(),
        name='leader_supter_evaluating',
        load_valid_test_loader=load_valid_test_loader,
        load_model=leader_supter_training.load_leader_supter_model,
        eval_model=eval_model)