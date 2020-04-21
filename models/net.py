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
from models.clf import Clf

import os
from scipy import io as sio

class Net(nn.Module):
    def __init__(self, featnet_1, refinenets, clf):
        super(Net, self).__init__()
        self.featnet_1 = featnet_1
        self.refinenets = refinenets
        self.clf = clf
        
    def forward(self, x):
        x = self.featnet_1(x)
        x = self.refinenets(x)
        x = self.clf(x)
        return x

def net(weights_dir=None, **kwargs):
    featnet_1 = Featnet_1()
    refinenets = Refinenets()
    clf = Clf()
    if weights_dir:
        featnet_1.load_state_dict(torch.load(os.path.join(weights_dir, 'featnet_1.pth')))
        refinenets.load_state_dict(torch.load(os.path.join(weights_dir, 'refinenets.pth')))
        clf.load_state_dict(torch.load(os.path.join(weights_dir, 'clf.pth')))
    return Net(featnet_1, refinenets, clf)