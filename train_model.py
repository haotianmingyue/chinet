

from chinet_model import CHINet
from torchvision.utils import save_image
from dic_loss import DiceLoss
from torch.utils.data import DataLoader

import torch
import torch.nn as nn


in_ch = 1
out_ch = 1

epoch = 100

device = 'cuda:0'

# 输入通道数和输出通道数
model  = CHINet(1, 1)

ce_loss = nn.CrossEntropyLoss()
dc_loss = DiceLoss()

alpha = 0.5
beta = 1 - alpha


optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.9,0.999), weight_decay=0.0001)


