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
from train import feature_regression_training

import wandb
import yaml
import json
import shutil

def arg_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--input-scale-factor', type=float, default=1.)
    parser.add_argument('--restore-run-id', type=str, required=True)

    parser.add_argument('--valid-batch-size', type=int, default=3)
    parser.add_argument('--test-batch-size', type=int, default=5)

    # parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--use-wandb', action='store_true')
    return parser

def load_valid_test_loader(args):
    cityscape_dataset_dir = '/home/user/data/Cityscapes'
    classes = ds.Cityscapes.classes
    id2label = {cls.id:cls for cls in classes}

    zurich_dataset_dir = '/home/user/data/Foggy_Zurich'

    def semantic2sparse(semantic):
        sparse = np.vectorize(lambda x: id2label[x].train_id)(np.array(semantic))
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
    device = next(model.parameters()).device
    model.eval()
    eval_running_metrics = [runningScore(20) for i in range(5)]
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(valid_dl), total=len(valid_dl))
        for _, ((b_clear, ), (b_beta_005, b_beta_01, b_beta_02, ), (b_sparse, _, )) in pbar:
            for running_metrics, b_input in zip(eval_running_metrics[:4], [b_clear, b_beta_005, b_beta_01, b_beta_02, ]):
                b_sparse_pred = model(b_input.to(device))['interp'].argmax(1).cpu()
                running_metrics.update(b_sparse.numpy(), b_sparse_pred.numpy(), )
            pbar.set_description("Valid Epoch {:3d}".format(wandb_log.running_metrics_epoch_step))
        if wandb_log.use_wandb:
            for name, running_metrics in zip(['clear', 'beta_0.005', 'beta_0.01', 'beta_0.02', ], eval_running_metrics[:4]):
                wandb_log.running_metrics_epoch_log(name, running_metrics)

        pbar = tqdm.tqdm(enumerate(test_dl), total=len(test_dl))
        for _, (b_input, b_sparse, _, ) in pbar:
            b_sparse_pred = model(b_input.to(device))['interp'].argmax(1).cpu()
            eval_running_metrics[-1].update(b_sparse.numpy(), b_sparse_pred.numpy(), )
        if wandb_log.use_wandb:
            wandb_log.running_metrics_epoch_log('testv2', eval_running_metrics[-1])
            pbar.set_description("Test Epoch {:3d}".format(wandb_log.running_metrics_epoch_step))

    for name, running_metrics in zip(['clear', 'beta_0.005', 'beta_0.01', 'beta_0.02', 'testv2'], eval_running_metrics):
        metrics, per_class_IoU = running_metrics.get_scores()
        pbar.write("{} Evaluation Metrics={}".format(name, metrics))
        pbar.write("{} Evaluation per_class_IoU={}".format(name, per_class_IoU))
    return eval_running_metrics

def main(parser, name, load_valid_test_loader, load_feature_regression_model, eval_model):
    args = parser.parse_args()
    print(args)
    if args.use_wandb:
        wandb.init(project='refinenet-pytorch', name=name, config=args, dir='/home/user/research/refinenet-pytorch/train')

    run_path = 'yonyeoseok/refinenet-pytorch/{}'.format(args.restore_run_id)
    print(run_path)
    with wandb.restore('wandb-metadata.json', run_path=run_path, root='.') as f:
        metadata = json.load(f)
        assert metadata['name'] == 'feature-regression-training', metadata['name']
        os.remove('wandb-metadata.json')
    with wandb.restore('config.yaml', run_path=run_path, root='.') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
        args.feature_layer = config['feature_layer']
        os.remove('config.yaml')

    valid_dl, test_dl = load_valid_test_loader(args)
    # valid_dl.dataset.indices = valid_dl.dataset.indices[:2]
    # test_dl.dataset.images = test_dl.dataset.images[:2]
    print('dataset loaded')
    feature_regression_model = load_feature_regression_model(args)
    print('model loaded')
    wandb_log = WandbLog(args.use_wandb)

    for epoch in range(int(config['total_epoch']['value'])+1):
        wandb_log.running_metrics_epoch_step = epoch
        state_dict_path = 'state_dict.{:02d}.pth'.format(epoch)
        if args.use_wandb:
            for local_run_dir in os.listdir(os.path.dirname(wandb.run.dir)):
                if args.restore_run_id in local_run_dir:
                    if state_dict_path in os.listdir(os.path.join(os.path.dirname(wandb.run.dir), local_run_dir)):
                        source_file = os.path.join(os.path.dirname(wandb.run.dir), local_run_dir, state_dict_path)
                        target_file = os.path.join(wandb.run.dir, state_dict_path)
                        shutil.copyfile(source_file, target_file)
                    break
        with wandb.restore(state_dict_path, run_path=run_path) as f:
            state_dict = torch.load(f.name)
        wandb.save(state_dict_path)
        feature_regression_model.load_state_dict(state_dict)
        eval_model(feature_regression_model.model, valid_dl, test_dl, wandb_log, args)
        if not args.use_wandb:
            os.remove(state_dict_path)

if __name__ == '__main__':
    main(
        parser=arg_parser(),
        name='feature_regression_evaluating',
        load_valid_test_loader=load_valid_test_loader,
        load_feature_regression_model=feature_regression_training.load_feature_regression_model,
        eval_model=eval_model)