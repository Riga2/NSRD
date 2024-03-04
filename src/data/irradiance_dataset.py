import os
import os.path as osp
import numpy as np
import torch
from data import data_utils
import random
from data.base_dataset import BaseDataset
import imageio
import time
import pickle
warped_save_dir = r'../experiment/warped_img'

def make_model(args, train=True):
    return IrradianceDataset(args, train)

class IrradianceDataset(BaseDataset):
    def __init__(self, args, train=True):
        super(IrradianceDataset, self).__init__(args)
        self.args = args
        self.train = train
        self.name = args.data_name
        self.crop_size = args.crop_size if train else None
        self.num_frames_samples = args.num_frames_samples if train else 1
        self.upsample = torch.nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=True)
        self.irradiance_dirname = args.irradiance_dirname

    def load_Irradiance_HR(self, folder_index, file_index, ext='.exr'):
        filename = str(file_index) + ext
        hr_file_path = os.path.join(self.gt_dir, str(folder_index), self.irradiance_dirname, filename)
        hr = data_utils.getFromExr(hr_file_path)
        return hr

    def load_Irradiance_LR(self, folder_index, file_index, ext='.exr'):
        filename = str(file_index) + ext
        lr_file_path = os.path.join(self.lr_dir, str(folder_index), self.irradiance_dirname, filename)
        lr = data_utils.getFromExr(lr_file_path)
        return lr

    def __getitem__(self, item):
        folder_index = item // self.grain
        file_index = item % self.grain

        HR_lst = []
        LR_lst = []
        MV_up_lst = []
        OCMV_up_lst = []
        Mask_up_lst = []

        lr_sh, lr_sw, hr_sh, hr_sw = 0, 0, 0, 0
        lr_eh, lr_ew = self.lr_size
        hr_eh, hr_ew = self.hr_size
        if self.crop_size is not None:
            ih, iw = self.lr_size
            tp = self.scale * self.crop_size
            ip = tp // self.scale
            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)
            tx, ty = self.scale * ix, self.scale * iy
            lr_sh, lr_sw, hr_sh, hr_sw = iy, ix, ty, tx
            lr_eh, lr_ew, hr_eh, hr_ew = iy+ip, ix+ip, ty+tp, tx+tp

        for index in range(file_index, file_index + self.num_frames_samples):
            # HR
            hr = self.load_Irradiance_HR(folder_index, index)
            # hdr -> ldr
            # hr = data_utils.toneMapping(hr)
            hr = data_utils.np2Tensor(hr)
            HR_lst.append(hr[:, hr_sh:hr_eh, hr_sw:hr_ew])

            # LR
            lr_files, mvs, ocmvs = [], [], []
            for idx in range(index, index - self.number_previous_frames - 1, -1):
                file = self.load_Irradiance_LR(folder_index, idx)
                # hdr -> ldr
                # file = data_utils.toneMapping(file)
                if self.useNormal:
                    normal = self.load_Normal_Unity(folder_index, idx)
                    file = np.concatenate((file, normal), axis=2)
                if self.useDepth:
                    depth = self.load_Depth_Unity(folder_index, idx)
                    file = np.concatenate((file, depth), axis=2)
                file = data_utils.np2Tensor(file)
                lr_files.append(file[:, lr_sh:lr_eh, lr_sw:lr_ew])

                # mv and ocmv
                if (idx != index - self.number_previous_frames):
                    mv = data_utils.np2Tensor(self.load_MV(folder_index, idx))
                    ocmv = data_utils.np2Tensor(self.load_OCMV(folder_index, idx))
                    mvs.append(mv[:, lr_sh:lr_eh, lr_sw:lr_ew])
                    ocmvs.append(ocmv[:, lr_sh:lr_eh, lr_sw:lr_ew])
                    if (idx == index):
                        mv_up = self.upsample(mv[None, :])[0]
                        ocmv_up = self.upsample(ocmv[None, :])[0]
                        mask_up = 1 - data_utils.mv_mask(ocmv_up, mv_up)[None, :]
                        MV_up_lst.append(mv_up[:, hr_sh:hr_eh, hr_sw:hr_ew])
                        OCMV_up_lst.append(ocmv_up[:, hr_sh:hr_eh, hr_sw:hr_ew])
                        Mask_up_lst.append(mask_up[:, hr_sh:hr_eh, hr_sw:hr_ew])

            # pre frames do Warp
            for i in range(self.number_previous_frames, 0, -1):
                for j in range(0, i-1):
                    mvs[i-1] += mvs[j]
                    ocmvs[i-1] += ocmvs[j]
                lr_files[i] = data_utils.backward_warp_motion(lr_files[i][None, :].cuda(), mvs[i-1][None, :].cuda(),
                                                            lr_files[0][None, :].cuda())[0].cpu()

            # pre frames cat mask
            for i in range(0, self.number_previous_frames):
                mask = data_utils.mv_mask(ocmvs[i], mvs[i])[None, :]
                lr_files[i+1] = torch.cat((lr_files[i+1], mask), dim=0)

            lr = torch.cat(lr_files, dim=0)
            LR_lst.append(lr)

        # for i in range(1, self.num_frames_samples-1):
        #     tmp_mv = MV_up_lst[i]
        #     tmp_ocmv = OCMV_up_lst[i]
        #     for j in range(0, i):
        #         tmp_mv += MV_up_lst[j]
        #         tmp_ocmv += OCMV_up_lst[i]
        #     tmp_mask = 1 - data_utils.mv_mask(tmp_ocmv, tmp_mv)[None, :]
        #     MV_up_lst.append(tmp_mv)
        #     Mask_up_lst.append(tmp_mask)

        # return LR_lst[0], HR_lst[0], str(file_index)
        return LR_lst, HR_lst, MV_up_lst, Mask_up_lst, str(file_index)

    def __len__(self):
        return self.total_folder_num * self.grain