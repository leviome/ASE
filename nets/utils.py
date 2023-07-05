from math import cos, pi, floor, sin
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio


def decimal(x, precise=6):
    base = pow(10, precise)
    return int(x * base) / base


def psnr(pred, target):
    func = PeakSignalNoiseRatio()
    return func(pred, target)


def loss_function(cam_code, image_code, recons, img, cam2img, bind_mode='mse'):
    recons_loss = F.mse_loss(recons, img)
    cam2img_loss = F.mse_loss(cam2img, img)
    if bind_mode == 'mse':
        cam_bind_loss = F.mse_loss(image_code, cam_code, reduction='sum')
    else:
        image_code = image_code.flatten(1)  # 4x625
        cam_code = cam_code.flatten(1)  # 4x625
        cam_bind_loss = 10000 * F.kl_div(image_code.softmax(-1).log(), cam_code.softmax(-1), reduction='sum')

    return recons_loss, cam_bind_loss, cam2img_loss


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cos(start, end, proportion):
    cos_val = cos(pi * proportion) + 1

    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = 0

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class CycleScheduler:
    def __init__(
            self,
            optimizer,
            lr_max,
            n_iter,
            momentum=(0.95, 0.85),
            divider=25,
            warmup_proportion=0.3,
            phase=('linear', 'cos'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {'linear': anneal_linear, 'cos': anneal_cos}

        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, phase_map[phase[1]]),
        ]

        self.momentum = momentum

        if momentum is not None:
            mom1, mom2 = momentum
            self.momentum_phase = [
                Phase(mom1, mom2, phase1, phase_map[phase[0]]),
                Phase(mom2, mom1, phase2, phase_map[phase[1]]),
            ]

        else:
            self.momentum_phase = []

        self.phase = 0

    def step(self):
        lr = self.lr_phase[self.phase].step()

        if self.momentum is not None:
            momentum = self.momentum_phase[self.phase].step()

        else:
            momentum = None

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                if 'betas' in group:
                    group['betas'] = (momentum, group['betas'][1])

                else:
                    group['momentum'] = momentum

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            for phase in self.momentum_phase:
                phase.reset()

            self.phase = 0

        return lr, momentum
