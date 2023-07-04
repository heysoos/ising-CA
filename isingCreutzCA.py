import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, HK_MAX, RADIUS=2):
        super().__init__()
        self.radius = RADIUS
        self.Hk_max = HK_MAX
        Rk = 2*RADIUS + 1

        nearest_neighbours = torch.zeros(1, 1, Rk, Rk).cuda()
        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0

        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)

    def forward(self, x):

        Rk = self.radius
        s = x[:, [0], ...]
        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')
        momentum = x[:, [1], ...]
        Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        delta_e = 2 * s * Js

        Hk_new = momentum - delta_e
        conserve_flag = torch.logical_and(Hk_new >= 0., Hk_new <= self.Hk_max)
        dropout_mask = (torch.rand_like(x[0, 0]) > 0.75).unsqueeze(0).unsqueeze(0)

        flip_flag = torch.logical_and(conserve_flag, dropout_mask)
        flip_spin = -2. * flip_flag + 1

        s_new = x[:, [0], ...] * flip_spin
        momentum = torch.where(flip_flag, Hk_new, momentum)

        return torch.cat([s_new, momentum], axis=1)

class isingCA(nn.Module):
    def __init__(self, HK_MAX, RADIUS=1):
        super().__init__()
        self.radius = RADIUS

        self.rule = Rule(HK_MAX, RADIUS)

    def initGrid(self, shape, init_order=None, init_energy=None):
        if init_energy is None:
            init_energy = self.rule.Hk_max
        if init_order is None:
            init_order = 0.5

        rand_spin = (torch.rand(1, 1, shape[0], shape[1]) > init_order) * 2. - 1.
        rand_momentum = torch.rand_like(rand_spin) * np.clip(init_energy, None, self.rule.Hk_max)

        return torch.cat([rand_spin, rand_momentum], axis=1).cuda()

    def forward(self, x):
        return self.rule(x)
