import sys
sys.path.append('/home/user/research/refinenet-pytorch')
import os
import numpy as np
import tqdm
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import datasets as ds
from torchvision import transforms as trf
# from torchvision.models._utils import IntermediateLayerGetter
# from models.refinenet_resnet import refinenet_resnet101

class FeatureRegressionModel(nn.ModuleDict):
    def __init__(self, model, target_model, feature_layer, cuda_mapping):
        model_named_children = list(model.named_children())
        target_model_named_children = list(target_model.named_children())
        assert list(zip(*model_named_children))[0] == list(zip(*target_model_named_children))[0]
        feature_layer_idx = list(zip(*model_named_children))[0].index(feature_layer)+1
        feature_model = nn.Sequential(OrderedDict(model_named_children[:feature_layer_idx]))
        target_feature_model = nn.Sequential(OrderedDict(target_model_named_children[:feature_layer_idx]))

        super(FeatureRegressionModel, self).__init__({
            'model': model,
            'feature_model': feature_model,
            'target_feature_model': target_feature_model})

        self['model'].to(cuda_mapping['model'])
        self['target_feature_model'].to(cuda_mapping['target_model'])
        self.cuda_mapping = cuda_mapping

    def forward(self, input):
        return (
            self['model'](input.to(self.cuda_mapping['model'])),
            self['feature_model'](input.to(self.cuda_mapping['model'])),
            self['target_feature_model'](input.to(self.cuda_mapping['target_model'])),)


class FeatureRegressionModelCriteria(nn.Module):
    def __init__(self, feature_regression_model, feature_regression_criteria, target_criteria, feature_regression_target_weight):
        super(FeatureRegressionModelCriteria, self).__init__()
        self.feature_regression_model = feature_regression_model
        self.feature_regression_criteria = feature_regression_criteria
        self.target_criteria = target_criteria
        # pylint: disable=E1101
        self.feature_regression_target_weight = torch.FloatTensor(feature_regression_target_weight)
        # pylint: enable=E1101

    def forward(self, input, target=None):
        logit, feature, target_feature = self.feature_regression_model(input)
        # pylint: disable=E1101
        feature_regression_target_loss = torch.stack([
            self.feature_regression_criteria(feature, target_feature.to(logit.device),),
            torch.FloatTensor(0) if target is None else
            self.target_criteria(logit, target.to(logit.device)),
        ])
        # pylint: enable=E1101
        return feature_regression_target_loss @ self.feature_regression_target_weight.to(feature_regression_target_loss.device)

    def state_dict(self):
        return self.feature_regression_model.state_dict()
