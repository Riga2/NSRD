import os
import math
import time
import datetime
import threading
from queue import Queue
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.ssim import ssim
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
from data.data_utils import save2Exr


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('ssim_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        if args.test_only:
            assert len(args.test_folder) == 1, "Currently only supports testing one folder at a time!"
            self.save_dir = 'sr_results_x{}/{}'.format(args.scale, args.test_folder[0])
            os.makedirs(self.get_path(self.save_dir), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 4

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_ssim(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('ssim_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_name)
        fig = plt.figure()
        plt.title(label)
        plt.plot(
            axis,
            self.log[:].numpy(),
            label='Scale {}'.format(self.args.scale)
        )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(self.get_path('test_{}.pdf'.format(self.args.data_name)))
        plt.close(fig)

    def plot_ssim(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_name)
        fig = plt.figure()
        plt.title(label)
        plt.plot(
            axis,
            (self.log[:]).numpy(),
            label='Scale {}'.format(self.args.scale)
        )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.savefig(self.get_path('test_{}.pdf'.format(self.args.data_name)))
        plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    im_np = tensor.numpy()
                    save2Exr(im_np, filename)
                    # im_np.tofile(filename)
                    # imageio.imwrite(filename, tensor.numpy())

        self.process = [
            threading.Thread(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                self.save_dir,
                '{}'.format(filename)
            )

            if self.args.sr_content == 'View':
                for v in save_list:
                    tensor_cpu = v[0].permute(1, 2, 0).cpu() * 255.0
                    self.queue.put(('{}.png'.format(filename), tensor_cpu))
            else:
                for v in save_list:
                    tensor_cpu = v[0].permute(1, 2, 0).cpu()
                    self.queue.put(('{}.exr'.format(filename), tensor_cpu))


def quantize(tensor):
    return torch.clamp(tensor, min=0.0)


def quantize_img(tensor):
    return torch.clamp(tensor, min=0.0, max=1.0)


def hdr2ldr(tensor):
    def adjust(data):
        contrast = 0.0
        brightness = 0.0
        contrastFactor = (259.0 * (contrast * 256.0 + 255.0)) / (255.0 * (259.0 - 256.0 * contrast))
        data = (data - 0.5) * contrastFactor + 0.5 + brightness
        return data

    chromatic = 0.5 * tensor
    adj = adjust(chromatic)
    tonemapping = adj / (adj + 1.0)
    gamma = torch.pow(tonemapping, 1.0 / 2.2)
    return gamma


def calc_psnr(sr, hr, scale, rgb_range):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def calc_mse(sr, hr):
    return F.mse_loss(sr, hr)


def calc_ssim(sr, hr):
    return ssim(sr, hr)


def calc_tempory(warped_sr, merge_sr, noc_mask):
    return F.l1_loss(warped_sr * noc_mask, merge_sr * noc_mask)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer
