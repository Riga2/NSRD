
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, conv_index):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
