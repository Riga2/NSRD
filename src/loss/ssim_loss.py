from model import ssim
import torch.nn as nn


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, sr, hr):
        return 1 - ssim.ssim(sr, hr)
