import configargparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm

def getFromExr(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def gamma_correct(data):
    data = np.where(data < 0.0031, data * 12.92, np.power(data, 1.0/2.4) * 1.055 - 0.055)
    return data

def sRGB(linear):
    return np.where(linear < 0.0031308,
                    12.92 * linear,
                    1.055 * np.power(linear, 1/2.4) - 0.055)

def toneMapTev(color):
    color[:, :, 0] = sRGB(color[:, :, 0])
    color[:, :, 1] = sRGB(color[:, :, 1])
    color[:, :, 2] = sRGB(color[:, :, 2])

    return np.clip(color, 0.0, 1.0)


def calc_psnr_ssim(res_dir, gt_dir):
    total_ssim = 0
    total_psnr = 0
    count = 0
    for name in tqdm(os.listdir(res_dir)):
        png_path = os.path.join(res_dir, name)
        gt_path = os.path.join(gt_dir, name)

        res_png = getFromExr(png_path)
        gt_png = getFromExr(gt_path)

        total_psnr += psnr(gt_png, res_png)
        total_ssim += ssim(gt_png, res_png, multichannel=True)
        count += 1

    print("avg_psnr : {}".format(total_psnr / count))
    print("avg_ssim : {}".format(total_ssim / count))

def remodulation(exp_dir, gt_dir):
    avg_psnr, avg_ssim = 0, 0
    sr_res_dir = os.path.join(exp_dir, 'sr_results_x4')
    img_save_dir = os.path.join(exp_dir, 'final_results_x4')
    folder_num = len(os.listdir(sr_res_dir))
    for ind in range(folder_num):
        os.makedirs(os.path.join(img_save_dir, f'{ind}'), exist_ok=True)

        cur_res_dir = os.path.join(sr_res_dir, f'{ind}')
        cur_res_lst = os.listdir(cur_res_dir)
        cur_num = len(cur_res_lst)
        cur_psnr, cur_ssim = 0, 0
        for name in tqdm(cur_res_lst):
            res_path = os.path.join(cur_res_dir, name)
            png_name = name.split('.')[0] + '.png'

            irr = getFromExr(res_path)
            irr[irr < 0] = 0
            brdf = getFromExr(os.path.join(gt_dir, f'{ind}', "BRDF", name))
            emiss_sky = getFromExr(os.path.join(gt_dir, f'{ind}', 'Emission_Sky', name))
            emiss_sky_mask = ((abs(emiss_sky[:, :, 0]) >= 1e-4) | (abs(emiss_sky[:, :, 1]) >= 1e-4) | (abs(emiss_sky[:, :, 2]) >= 1e-4))[:, :, np.newaxis]

            sr_img = brdf * irr
            sr_img = np.where(emiss_sky_mask, emiss_sky, sr_img)
            sr_img = (toneMapTev(sr_img)*255).astype(np.uint8)

            gt_img = getFromExr(os.path.join(gt_dir, f'{ind}', "View_PNG", png_name))
            cur_psnr += psnr(gt_img, sr_img)
            cur_ssim += ssim(gt_img, sr_img, win_size=11, channel_axis=2, data_range=255)

            save_path = os.path.join(cur_res_dir, png_name)
            cv2.imwrite(save_path, sr_img[:, :, ::-1])

        cur_psnr /= cur_num
        cur_ssim /= cur_num
        avg_psnr += cur_psnr
        avg_ssim += cur_ssim

    avg_psnr /= folder_num
    avg_ssim /= folder_num

    print("Avg_pnsr: {}".format(avg_psnr))
    print("Avg_ssim: {}".format(avg_ssim))


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default=r"../experiment/Bistro_X4",
                        help='experiment dir')
    parser.add_argument('--gt_dir', type=str, default=r"../dataset/Bistro/test/GT",
                        help='ground truth dir, which contains View_PNG, BRDF and Emisson_Sky.')
    args = parser.parse_args()

    remodulation(args.exp_dir, args.gt_dir)


