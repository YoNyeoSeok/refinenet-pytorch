"""RefineNet

RefineNet PyTorch for non-commercial purposes

Copyright (c) 2018, YoNyeoSeok (yys8646@postech.ac.kr)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn

from models.resnet import (Resnet101, resnet101)
from models.refinenets import Refinenets, refinenets
from models.clf import Clf, clf

import os

from collections import OrderedDict


class RefineNet_ResNet(nn.Sequential):
    def __init__(self, resnet=Resnet101(), refinenets=Refinenets(), clf=Clf()):
        super(RefineNet_ResNet, self).__init__(
            OrderedDict([
                ('resnet', resnet),
                ('refinenets', refinenets),
                ('clf', clf)
            ]))
    
    def forward(self, input):
        for module in self:
            device = next(module.parameters()).device
            if isinstance(input, (tuple, list)):
                input = tuple([inp.to(device) for inp in input])
            elif isinstance(input, torch.Tensor):
                input = input.to(device)
            input = module(input)
        return input
    

def refinenet_resnet101(weights_dir=None, **kwargs):
    if weights_dir:
        model = RefineNet_ResNet(
            resnet101(os.path.join(weights_dir, 'resnet101.pth')),
            refinenets(os.path.join(weights_dir, 'refinenets.pth')),
            clf(os.path.join(weights_dir, 'clf.pth'))) 
    else:
        model = RefineNet_ResNet()
    return model