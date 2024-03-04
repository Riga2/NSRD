import torch
import torch.nn as nn
from torch.nn import functional as F

class Temporal_Loss(nn.Module):
    def __init__(self):
        super(Temporal_Loss, self).__init__()

    def forward(self, sr_pre, sr_cur):
        return F.smooth_l1_loss(sr_pre, sr_cur)
