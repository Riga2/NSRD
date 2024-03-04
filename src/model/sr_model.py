import torch.nn as nn
import torch
from model import sr_common as sr_com
import torch.nn.functional as F
from model.ConvLSTM import ConvLSTM

def make_model(args):
    if args.num_pre_frames != 0:
        return SRNet(args)

class SRNet(nn.Module):
    def __init__(self, args=None, conv=sr_com.default_conv):
        super(SRNet, self).__init__()
        self.scale = args.scale
        self.act = nn.ReLU(True)
        self.in_dims = args.input_total_dims
        self.n_feats = args.n_feats
        self.num_previous = args.num_pre_frames
        self.total_feats = self.n_feats * (self.num_previous + 1) + self.n_feats

        self.conv1 = conv(self.in_dims, self.n_feats, 3)
        self.gate_conv = sr_com.GatedConv2dWithActivation(self.in_dims + 1, self.n_feats, 3, padding=1)

        feats = 64
        self.unps = nn.PixelUnshuffle(self.scale)
        self.conv2 = conv(3 * self.scale * self.scale, self.n_feats, 3)
        self.convLSTM = ConvLSTM(self.total_feats, feats, kernel_size=3)

        # U-shaped reconstruction module
        self.encoder_1 = nn.Sequential(
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16)
        )

        self.encoder_2 = nn.Sequential(
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16)
        )

        self.center = nn.Sequential(
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16)
        )

        self.decoder_2 = nn.Sequential(
            conv(feats * 2, feats, 3),
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16)
        )

        self.decoder_1 = nn.Sequential(
            conv(feats * 2, feats, 3),
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16),
            sr_com.RCAB(conv, feats, 3, 16)
        )

        self.pooling = nn.MaxPool2d(2)
        self.upsize = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = conv(feats, self.n_feats, 3)

        self.upsampling = nn.Sequential(
            conv(self.n_feats, args.output_dims * self.scale * self.scale, 3),
            nn.PixelShuffle(self.scale)
        )

    def crop_tensor(self, actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        diffY = int(actual.size()[2] - target.size()[2])
        diffX = int(actual.size()[3] - target.size()[3])
        x = F.pad(target, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, x_tuple):
        x, last_sr, prev_state = x_tuple

        # split cur frames
        x_cur_frames = x[:, :self.in_dims, :, :]
        x_cur_frames = self.conv1(x_cur_frames)

        # the last channel of previous frame is mask
        x_pre_frames = []
        ch_pre_frames = self.in_dims + 1

        # split previous frames
        for i in range(0, self.num_previous):
            st, ed = self.in_dims + i * ch_pre_frames, self.in_dims + (i + 1) * ch_pre_frames
            pre_frame = x[:, st:ed, :, :]
            pre_frame = self.gate_conv(pre_frame)
            x_pre_frames.append(pre_frame)
        x_pre_frame = torch.cat(x_pre_frames, dim=1)

        # last sr
        last_ups = self.unps(last_sr)
        last_in = self.conv2(last_ups)

        # path both
        x_all = torch.cat((x_cur_frames, x_pre_frame, last_in), dim=1)
        state = self.convLSTM(x_all, prev_state)

        x_encoder1 = self.encoder_1(state[0])
        x_encoder1_pool = self.pooling(x_encoder1)
        x_encoder2 = self.encoder_2(x_encoder1_pool)
        x_encoder2_pool = self.pooling(x_encoder2)

        x_center = self.center(x_encoder2_pool)

        x_center_up = self.crop_tensor(x_encoder2, self.upsize(x_center))
        x_decoder2 = self.decoder_2(torch.cat((x_center_up, x_encoder2), dim=1))

        x_decoder2_up = self.crop_tensor(x_encoder1, self.upsize(x_decoder2))
        x_decoder1 = self.decoder_1(torch.cat((x_decoder2_up, x_encoder1), dim=1))

        x_in = self.conv3(x_decoder1)
        x_res = self.upsampling(x_in)

        return x_res, state

