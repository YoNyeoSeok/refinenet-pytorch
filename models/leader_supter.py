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

class LeaderSupterModel(nn.ModuleDict):
    def __init__(
        self, leader_model, supter_model,
        cuda_mapping={
            'leader_model': 'cpu',
            'supter_model': 'cpu',}):
        super(LeaderSupterModel, self).__init__({
            'leader_model': leader_model,
            'supter_model': supter_model,})
        assert self['leader_model'] != self['supter_model']
        self['leader_model'].to(cuda_mapping['leader_model'])
        self['supter_model'].to(cuda_mapping['supter_model'])
        self['supter_model'].eval()
        self.cuda_mapping = cuda_mapping
    def forward(self, input):
        return (
            self['leader_model'](input.to(self.cuda_mapping['leader_model'])),
            self['supter_model'](input.to(self.cuda_mapping['supter_model'])),)
    def update_supter(self):
        self.supter_model.load_state_dict(self.leader_model.state_dict()).eval()
    def train(self, mode=True):
        self.leader_model.train(mode)
    def eval(self):
        self.leader_model.eval()

class LeaderSupterModelCriteria(nn.Module):
    def __init__(self, leader_supter_model, supter_criteria, target_criteria, supter_target_weight):
        super(LeaderSupterModelCriteria, self).__init__()
        self.leader_supter_model = leader_supter_model
        self.supter_criteria = supter_criteria
        self.target_criteria = target_criteria
        # pylint: disable=E1101
        self.supter_target_weight = torch.FloatTensor(supter_target_weight)
        # pylint: enable=E1101

    def forward(self, input, target=None):
        leader_logit, supter_logit = self.leader_supter_model(input)
        # pylint: disable=E1101
        supter_target_loss = torch.stack([
            self.supter_criteria(leader_logit, supter_logit.to(leader_logit.device),),
            torch.FloatTensor(0) if target is None else
            self.target_criteria(leader_logit, target.to(leader_logit.device)),
        ])
        # pylint: enable=E1101
        return supter_target_loss @ self.supter_target_weight.to(supter_target_loss.device)

    def state_dict(self):
        return self.leader_supter_model.state_dict()

class CudaInputLayer(nn.Module):
    def __init__(self, ):
        super(CudaInputLayer)

class CudaChildrenLayer(nn.Sequential):
    def __init__(self, mapping_dict, *args, **kwds):
        super(CudaChildrenLayer, self, ).__init__(*args, **kwds)
        assert set([name for name, _ in self.named_children()]) == set(list(mapping_dict.keys())) 
        # self.model = model
        self.mapping_dict = mapping_dict
        # for name, child in self.named_children():
        #     self.__setattr__(name, CudaChildrenLayer(
        #         {k: mapping_dict[name] for k, _ in child.named_children()},
        #         OrderedDict([(k, v) for k, v in child.named_children()])))
        #     assert name != model
        #     self.__setattr__(name, child)
    def forward(self, input):
        x = input
        if isinstance(self, LeaderSupterModel):
            self['LeaderSupterModel'] = CudaChildrenLayer(self['LeaderSupterModel'])
        for name, child in self.named_children():
            x = child(x.to(self.mapping_dict[name]))
        return x
    def state_dict(self):
        return self.model.state_dict()

class ReturnChildLayer(nn.Module):
    def __init__(self, *args, **kwds):
        super(ReturnChildLayer, self).__init__(*args, **kwds)
    def forward(self, input, return_name=None):
        if return_name is not None:
            assert return_name in [name for name in self.named_children()]
        x = input
        for name, child in self.named_children():
            x = child(x)
            if name == return_name:
                return x


# class ReturnChildLayerLeaderSupter(nn.Module):
#     def __init__(self, supter_model_dict, leader_model_dict, intermediate_layer):
#         super(ReturnChildLayerLeaderSupter, self).__init__()
#         self.supter_model_dict = supter_model_dict
#         self.leader_model_dict = leader_model_dict
#         self.intermediate_layer = intermediate_layer
#     def forward(self, input, return_key=None):
#         return self.leader_model_dict(input, return_key)
#     def poo(self, input, criteria, return_key=None):
#         return criteria(self.leader_model(input, return_key), self.supter_model(input, return_key))
#     def bar(self, input, target, criteria, return_key=None):
#         return criteria(self.leader_model(input, return_key), target)
#     def update_supter(self, return_key=None):
#         for (name, leader_child), supter_child in zip(self.leader_model_dict.items(), self.supter_model_dict.values()):
#             supter_child.load_state_dict(leader_child.state_dict())
#             if name == return_key:
#                 break
#     def train(self): 
#         self.leader_model_dict.train()
#     def eval(self):
#         self.leader_model_dict.eval()
