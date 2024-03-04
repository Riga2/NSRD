import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import torch
import torch.nn as nn
import random
import imageio
import torch.nn.functional as F
import time

def linearUpsample(img, scale):
    return cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def save2Exr(img, path):
    cv2.imwrite(path, img[:,:,::-1])

def getFromBin(path, ih, iw):
    return np.fromfile(path, dtype=np.float32).reshape(ih, iw, -1).transpose(0, 1, 2)

def getFromExr(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        print(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def bin2exr(pt_path, ih, iw, save_path):
    irr = getFromBin(pt_path, ih, iw)[:, :, ::-1]
    cv2.imwrite(save_path, irr)

def get_patch(lr, hr, patch_size=256, scale=2):
    ih, iw = lr.shape[1:]
    tp = scale * patch_size
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy
    ret = [
        lr[:, iy:iy + ip, ix:ix + ip],
        hr[:, ty:ty + tp, tx:tx + tp]
    ]
    return ret

def np2Tensor(np_file):
    return torch.from_numpy(np_file).permute(2, 0, 1).float()

def tensorFromNp(np_file):
    np_file = np_file.transpose(2, 0, 1)
    return torch.FloatTensor(np_file)

def mv_mask(ocmv, mv, gate = 0.1):
    delta = ocmv - mv
    mask = torch.where(torch.abs(delta) < gate, False, True)
    x, y = mask[0, :, :], mask[1, :, :]
    mask = torch.where(((x) | (y)), 1, 0)
    return mask

def backward_warp_motion(pre: torch.Tensor, motion: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    # see: https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
    # input image is: [batch, channel, height, width]
    # st = time.time()
    index_batch, number_channels, height, width = pre.size()
    grid_x = torch.arange(width).view(1, -1).repeat(height, 1)
    grid_y = torch.arange(height).view(-1, 1).repeat(1, width)
    grid_x = grid_x.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    grid_y = grid_y.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    #
    grid = torch.cat((grid_x, grid_y), 1).float().cuda()
    # grid is: [batch, channel (2), height, width]
    vgrid = grid - motion
    # Grid values must be normalised positions in [-1, 1]
    vgrid_x = vgrid[:, 0, :, :]
    vgrid_y = vgrid[:, 1, :, :]
    vgrid[:, 0, :, :] = (vgrid_x / width) * 2.0 - 1.0
    vgrid[:, 1, :, :] = (vgrid_y / height) * 2.0 - 1.0
    # swapping grid dimensions in order to match the input of grid_sample.
    # that is: [batch, output_height, output_width, grid_pos (2)]
    vgrid = vgrid.permute((0, 2, 3, 1))
    warped = F.grid_sample(pre, vgrid, align_corners=True)

    # return warped
    oox, ooy = torch.split((vgrid < -1) | (vgrid > 1), 1, dim=3)
    oo = (oox | ooy).permute(0, 3, 1, 2)
    # ed = time.time()
    # print('warp {}'.format(ed-st))
    return torch.where(oo, cur, warped)

def warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid - flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default for PyTorch version < 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output

def toneMapping(data):
    return data / (data + 1.0)

def ldr2hdr(data):
    return data / (1.0 - data)

def chromatic(data):
    return 0.5 * data

def adjust(data):
    contrast = 0.0
    brightness = 0.0
    contrastFactor = (259.0 * (contrast * 256.0 + 255.0)) / (255.0 * (259.0 - 256.0 * contrast))
    data = (data - 0.5) * contrastFactor + 0.5 + brightness
    return data


def gamma_correct(data):
    r = 1.0 / 2.2
    return np.power(data, r)

