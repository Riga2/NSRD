import configargparse

parser = configargparse.ArgumentParser(description='NSRD')

# Config file
parser.add_argument('--config', is_config_file=True, default='../configs/Bistro_train.txt',
                    help='config file path')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--root_dir', type=str, default=r"../dataset/Bistro/train",
                    help='rendering data_dir')
parser.add_argument('--data_name', type=str, default='Bistro',
                    help='test dataset name')
parser.add_argument('--view_dirname', type=str, default='View',
                    help='view dirname')
parser.add_argument('--irradiance_dirname', type=str, default='Irradiance',
                    help='Irradiance dirname')
parser.add_argument('--depth_dirname', type=str, default='Depth',
                    help='depth dirname')
parser.add_argument('--normal_dirname', type=str, default='Normal',
                    help='normal dirname')
parser.add_argument('--mv_dirname', type=str, default='MV',
                    help='LR mv dirname')
parser.add_argument('--ocmv_dirname', type=str, default='OCMV',
                    help='LR ocmv dirname')
parser.add_argument('--brdf_dirname', type=str, default='BRDF',
                    help='LR ocmv dirname')

parser.add_argument('--use_normal', type=bool, default=True,
                    help='use normal')
parser.add_argument('--use_depth', type=bool, default=True,
                    help='use depth')
parser.add_argument('--num_frames_samples', type=int, default=4,
                    help='number of previous frames')
parser.add_argument('--num_pre_frames', type=int, default=2,
                    help='number of previous frames')
parser.add_argument('--input_total_dims', type=int, default=3+1+3,
                    help='input total dims')
parser.add_argument('--output_dims', type=int, default=3,
                    help='output dims')
parser.add_argument('--sr_content', type=str, default='Irradiance',
                    help='choose which content to super resolution, '
                         '(Irradiance, View, LMDB)')

parser.add_argument('--gt_size', type=tuple, default=(1080, 1920),
                    help='GT size(ih, iw)')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--crop_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--grain', type=int, default=100,
                    help='file number in a folder')
parser.add_argument('--total_folder_num', type=int, default=60,
                    help='total folder num')
parser.add_argument('--test_folder', type=int, action="append",
                    help='test folder index')
parser.add_argument('--valid_folder_num', type=int, default=6,
                    help='valid folder number')

# Model specifications
parser.add_argument('--model', default='sr_model',
                    help='model name')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_feats', type=int, default=32,
                    help='number of feature maps')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true', default=False,
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='100',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1.0*L1+1.0*SSIM+1.0*Temporal',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='Bistro_X4',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=1,
                    help='resume from specific checkpoint: -1 -> best, 0 -> latest, else -> None')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=False,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


