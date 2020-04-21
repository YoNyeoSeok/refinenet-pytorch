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

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from models.featnet_1 import Featnet_1
from models.refinenets import Refinenets

import os
from scipy import io as sio

class Net(nn.Module):
    def __init__(self, featnet_1, refinenets, rate, num_classes):
        super(Net, self).__init__()
        self.featnet_1 = featnet_1
        self.refinenets = refinenets
        self.do = nn.Dropout(p=rate)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        x = self.featnet_1(x)
        x = self.refinenets(x)
        x = self.do(x)
        x = self.clf_conv(x)
        return x

def net(weights_dir=None, pretrained=True):
    featnet_1 = Featnet_1()
    refinenets = Refinenets()
    if pretrained:
        assert weights_dir != None
        featnet_1.load_state_dict(torch.load(os.path.join(weights_dir, 'featnet_1.pth')))
        refinenets.load_state_dict(torch.load(os.path.join(weights_dir, 'refinenets.pth')))

    model = Net(featnet_1, refinenets, rate=0.5, num_classes=20)
    if pretrained:
        model.clf_conv.load_state_dict(torch.load(os.path.join(weights_dir, 'clf.pth')))
    return model