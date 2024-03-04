import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from torch.utils.data import Dataset
from data import data_utils

class BaseDataset(Dataset):
    def __init__(self, args=None):
        super(BaseDataset, self).__init__()
        self.args = args
        self.scale = args.scale
        self.number_previous_frames = args.num_pre_frames
        self.hr_size = args.gt_size
        self.lr_size = (args.gt_size[0] // self.scale, args.gt_size[1] // self.scale)

        self.gt_dir = os.path.join(args.root_dir, 'GT')
        self.lr_dir = os.path.join(args.root_dir, 'X' + str(args.scale))

        self.useNormal, self.useDepth = args.use_normal, args.use_depth
        self.depth_dirname = args.depth_dirname
        self.normal_dirname = args.normal_dirname
        self.mv_dirname = args.mv_dirname
        self.ocmv_dirname = args.ocmv_dirname
        self.grain = args.grain
        self.total_folder_num = args.total_folder_num

    def load_Normal_Unity(self, folder_index, file_index, ext='.exr'):
        filename = str(file_index) + ext
        lr_file_path = os.path.join(self.lr_dir, str(folder_index), self.normal_dirname, filename)
        lr = data_utils.getFromExr(lr_file_path)[:, :, :3]
        return lr

    def load_Depth_Unity(self, folder_index, file_index, ext='.exr'):
        filename = str(file_index) + ext
        lr_file_path = os.path.join(self.lr_dir, str(folder_index), self.depth_dirname, filename)
        lr = data_utils.getFromExr(lr_file_path)[:, :, 0][:, :, None]
        return lr

    def load_MV(self, folder_index, file_index, ext='.exr'):
        filename = str(file_index) + ext
        lr_file_path = os.path.join(self.lr_dir, str(folder_index), self.mv_dirname, filename)
        # lr = data_utils.getFromBin(lr_file_path, self.lr_size[0], self.lr_size[1])[:, :, :2]
        lr = data_utils.getFromExr(lr_file_path)[:, :, :2]
        lr[:, :, 0] = lr[:, :, 0] * self.lr_size[1]
        lr[:, :, 1] = lr[:, :, 1] * self.lr_size[0]
        return lr

    def load_OCMV(self, folder_index, file_index, ext='.exr'):
        filename = str(file_index) + ext
        lr_file_path = os.path.join(self.lr_dir, str(folder_index), self.ocmv_dirname, filename)
        # lr = data_utils.getFromBin(lr_file_path, self.lr_size[0], self.lr_size[1])[:, :, :2]
        lr = data_utils.getFromExr(lr_file_path)[:, :, :2]
        lr[:, :, 0] = lr[:, :, 0] * self.lr_size[1]
        lr[:, :, 1] = lr[:, :, 1] * self.lr_size[0]
        return lr

