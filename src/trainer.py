import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import time
from decimal import Decimal
import numpy as np
import imageio
import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from data import data_utils

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer, self).__init__()
        self.args = args
        self.scale = args.scale
        self.gt_size = args.gt_size
        self.batch_size = args.batch_size
        self.ckp = ckp
        self.model = my_model
        self.num_frames_samples = args.num_frames_samples
        self.train_loader = loader.loader_train
        self.valid_loader = loader.loader_valid
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        for batch, (LR_lst, HR_lst, MV_up_lst, Mask_up_lst, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            b, c, h, w = HR_lst[0].size()
            zero_tensor = torch.zeros(b, c, h, w, dtype=torch.float32)
            lr0, zero_tensor, hr0 = self.prepare(LR_lst[0], zero_tensor, HR_lst[0])

            sr_pre, lstm_state = self.model((lr0, zero_tensor, None))
            lstm_state = utility.repackage_hidden(lstm_state)
            loss = self.loss(sr_pre, hr0, needTem=False)

            for i in range(1, self.num_frames_samples):
                sr_pre = sr_pre.detach()
                sr_pre.requires_grad = False

                lr, hr, mv_up, mask_up = self.prepare(LR_lst[i], HR_lst[i], MV_up_lst[i], Mask_up_lst[i])

                timer_data.hold()
                timer_model.tic()

                sr_pre_warped = data_utils.warp(sr_pre, mv_up)
                sr_cur, lstm_state = self.model((lr, sr_pre_warped, lstm_state))
                lstm_state = utility.repackage_hidden(lstm_state)

                loss += self.loss(sr_cur, hr, sr_pre_warped, mask_up, needTem=True)
                sr_pre = sr_cur
            loss.backward()
            self.optimizer.step()
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    self.train_loader.n_samples,
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, 3)
        )
        self.model.eval()

        timer_test = utility.timer()
        run_model_time = 0
        flag = 0
        if self.args.save_results: self.ckp.begin_background()

        pre_sr = torch.zeros(1, 3, self.gt_size[0], self.gt_size[1],
                             dtype=torch.float32).cuda()
        lstm_state = None
        for index, (LR_lst, HR_lst, MV_up_lst, Mask_up_lst, filename) in tqdm(enumerate(self.valid_loader)):
            lr, hr, mv_up, mask_up = self.prepare(LR_lst[0], HR_lst[0], MV_up_lst[0], Mask_up_lst[0])
            if index == 0:
                pre_sr, lstm_state = self.model((lr, pre_sr, lstm_state))
                lstm_state = utility.repackage_hidden(lstm_state)
                continue
            t1 = time.time()
            sr_pre_warped = data_utils.warp(pre_sr, mv_up)
            cur_sr, lstm_state = self.model((lr, sr_pre_warped, lstm_state))
            lstm_state = utility.repackage_hidden(lstm_state)
            t2 = time.time()
            run_model_time += (t2 - t1)
            if self.args.sr_content == "View":
                sr = utility.quantize_img(cur_sr)
                sr_last = utility.quantize_img(pre_sr)
                if flag < 2:
                    data_utils.save2Exr(np.array(sr[0, :3, :, :].permute(1, 2, 0).detach().cpu()) * 255,
                                        ".\\check\\sr_" + str(flag) + ".png")
                    data_utils.save2Exr(np.array(hr[0, :3, :, :].permute(1, 2, 0).detach().cpu()) * 255,
                                        ".\\check\\gt_" + str(flag) + ".png")
                    flag += 1
            else:
                sr = utility.quantize(cur_sr)
                sr_last = utility.quantize(pre_sr)
                if flag < 2:
                    data_utils.save2Exr(np.array(sr[0, :3, :, :].permute(1, 2, 0).detach().cpu()),
                                        ".\\check\\sr_" + str(flag) + ".exr")
                    data_utils.save2Exr(np.array(hr[0, :3, :, :].permute(1, 2, 0).detach().cpu()),
                                        ".\\check\\gt_" + str(flag) + ".exr")
                    flag += 1

            pre_sr = cur_sr
            save_list = [sr]
            assert sr is not torch.nan, "sr is nan!"
            val_ssim = 1.0 - utility.calc_ssim(sr, hr).cpu()
            warped_sr = data_utils.warp(sr_last, mv_up)
            val_tempory = utility.calc_tempory(warped_sr, sr, mask_up).cpu()

            self.ckp.log[-1, 0] += val_ssim
            self.ckp.log[-1, 1] += val_tempory
            self.ckp.log[-1, 2] += val_tempory + val_ssim

            if self.args.save_gt:
                save_list.extend([lr, hr])

            if self.args.save_results:
                self.ckp.save_results(self.valid_loader, filename[0], save_list, self.scale)

        self.ckp.log[-1] /= (len(self.valid_loader) - 1)
        best = self.ckp.log.min(0)
        self.ckp.write_log(
            '[{} x{}]\tSSIM: {:.6f}, Tempory: {:.6f}, Total :{:.6f} (Best: {:.6f} @epoch {})'.format(
                self.valid_loader.dataset.name,
                self.scale,
                self.ckp.log[-1][0],
                self.ckp.log[-1][1],
                self.ckp.log[-1][2],
                best[0][2],
                best[1][2] + 1
            )
        )

        self.ckp.write_log('Run model time {:.5f}s\n'.format(run_model_time))
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[0][2] is not torch.nan and best[1][2] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs