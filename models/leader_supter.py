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
# from torchvision.models._utils import IntermediateLayerGetter
# from models.refinenet_resnet import refinenet_resnet101

class LeaderSupterModel(nn.Module):
    def __init__(self, leader_model, supter_model):
        super(LeaderSupterModel, self).__init__()
        assert leader_model != supter_model
        self.leader_model = leader_model
        self.supter_model = supter_model
    def forward(self, input):
        return self.leader_model(input)
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
        # pylint: disable=E1101
        supter_target_loss = torch.stack([
            self.supter_criteria(
                self.leader_supter_model.leader_model(input),
                self.leader_supter_model.supter_model(input),),
            torch.FloatTensor(0) if target is None else
            self.target_criteria(
                self.leader_supter_model.leader_model(input), target),
        ])
        # pylint: enable=E1101
        return supter_target_loss @ self.supter_target_weight.to(supter_target_loss.device)

# class CudaChilderenLayer(nn.Module):
#     def __init__(self, mapping_dict, *args, **kwds):
#         super(CudaChilderenLayer, self, ).__init__(*args, **kwds)
#         assert set([name for name, _ in self.named_children()]) == set(list(mapping_dict.keys())) 
#         self.mapping_dict = mapping_dict
#     def forward(self, input):
#         x = input
#         for name, child in self.named_children():
#             x = child(x.to(self.mapping_dict[name]))
#         return x

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
