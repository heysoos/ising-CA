import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, RES, BETA=1, RADIUS=1):
        super().__init__()
        self.beta = BETA
        self.radius = RADIUS
        self.res = RES
        Rk = 2*RADIUS + 1

        self.J_adapt = False
        self.adapt_lr = 0.1
        self.alpha = 0.9
        self.h = 0.01
        self.eps = 0.01
        self.max_weight = 4.

        self.trace_memory = 0.995

        self.num_obvs = 4
        self.obv_decay = 0.95

        self.numel = RES[0] * RES[1]

        # radial kernel
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        exp_rm = torch.exp(-rm ** 2)
        condition = rm < 1.
        null = torch.zeros_like(rm).cuda()
        self.rm = torch.where(condition, exp_rm, null).unsqueeze(0).unsqueeze(-1)

        nearest_neighbours = torch.randn(1, Rk, Rk, self.numel).cuda()
        # nearest_neighbours = torch.zeros(1, Rk, Rk, self.numel).cuda()/ RADIUS ** 2

        # nearest_neighbours[:, RADIUS, :, :] = 1.
        # nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, RADIUS, RADIUS, :] = 0

        # self.nearest_neighbours = (nearest_neighbours * self.rm).reshape(1, -1, self.numel)
        self.nearest_neighbours = 1 * (nearest_neighbours).reshape(1, -1, self.numel)


    def forward(self, x, make_obv=False, zealots=None):

        s = x[:, [0], ...]
        tr = x[:, [1], ...]
        b = x[:, [-1], ...]
        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')

        sJs = 2 * F.unfold(s, 1) * (F.unfold(s_pad, 2 * Rk + 1) * self.nearest_neighbours)
        # Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        delta_e = sJs.sum(dim=1).view(*shape)
        E = (-0.5 * delta_e)

        Js = (F.unfold(s_pad, 2*Rk+1) * self.nearest_neighbours).sum(dim=1).view(*shape)
        delta_e = 2 * s * Js

        ### OBSERVATIONS ###
        if make_obv:
            if zealots is not None:
                e = E[~zealots].mean() / 4.

                m = s[~zealots].mean()
            else:
                e = E.mean() / 4.
                m = s.mean()

            e2 = e ** 2
            m2 = m ** 2
            obvs = torch.stack([e, e2, m, m2], axis=0)
            #####################

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * b)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        if self.J_adapt and torch.rand(1) > 0.5:
            if zealots is not None:
                zmask_flat = (F.unfold(1. * zealots, 1) > 0.).squeeze()
            s_i = F.unfold(tr, 1)
            s_j = F.unfold(F.pad(tr, (Rk, Rk, Rk, Rk), mode='circular'), 2*Rk+1)


            growth = self.h / Rk * (1 - s_i.abs()) * (1 - s_j.abs())  # correlate
            # decay = self.eps * (s_j.mean(dim=1, keepdim=True) * s_i)  # decorrelate if mag.
            decay = self.eps * (s_j * s_i)  # decorrelate if mag.
            dJ = (growth - decay)  # * self.rm.reshape(1, -1, 1)

            new_J = self.nearest_neighbours + \
                    dJ * self.adapt_lr * (torch.rand_like(self.nearest_neighbours) > 0.5) # stochastic dJ
            new_J[:, Rk * (2 * Rk + 1) + Rk, :] = 0.  # set center pixel to 0.

            # set corners to 0
            # new_J[:, 0, :] = 0.
            # new_J[:, 2*Rk, :] = 0.
            # new_J[:, -(2*Rk+1), :] = 0.
            # new_J[:, -1, :] = 0.

            # set zealots to 0
            if zealots is not None:
                new_J[:, :, zmask_flat] = 0.

            self.nearest_neighbours = ((1 - self.alpha) * self.nearest_neighbours + self.alpha * new_J)

        tr = self.trace_memory * tr + (1 - self.trace_memory) * s

        if make_obv:
            return torch.cat([(s * flip), tr, b], axis=1), obvs
        else:
            return torch.cat([(s * flip), tr, b], axis=1)


class isingCA(nn.Module):
    def __init__(self, RES, BETA=1, RADIUS=2):
        super().__init__()
        self.radius = RADIUS
        self.rule = Rule(RES, BETA, RADIUS)

    def initGrid(self, unfold=True):
        shape = self.rule.res
        rand = (torch.rand(1, 3, shape[0], shape[1]) > 0.5) * 2. - 1.
        rand[:, 1, ...] = torch.zeros_like(rand[:, 1, ...])
        rand[:, -1, ...] = torch.ones_like(rand[:, -1, ...]) * self.rule.beta

        return rand.cuda()

    def forward(self, x, make_obv=False, zealots=None):
        return self.rule(x, make_obv, zealots)
