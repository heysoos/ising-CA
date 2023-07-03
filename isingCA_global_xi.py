import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, RES, BETA=1, CHANNELS=8, RADIUS=1):
        super().__init__()
        self.res = RES
        self.channels = CHANNELS
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        self.numel = RES[0] * RES[1]

        self.num_obvs = 4

        nearest_neighbours = torch.zeros(1, Rk, Rk, self.numel).cuda()

        nearest_neighbours[:, RADIUS, :, :] = 1.
        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, RADIUS, RADIUS, :] = 0

        self.nearest_neighbours = (nearest_neighbours).reshape(1, -1, self.numel)

    def forward(self, x):

        s = x[:, :self.channels, ...]
        b = x[:, [-1], ...]

        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')
        # b_pad = F.pad(b, (Rk, Rk, Rk, Rk), mode='circular')

        sJs = 2 * F.unfold(s, 1) * (F.unfold(s_pad, 2 * Rk + 1) * self.nearest_neighbours)
        delta_e = sJs.sum(dim=1).view(*shape)
        E = (-0.5 * delta_e)

        ### OBSERVATIONS ###

        e = E.mean() / 4.
        e2 = e**2

        m = s.mean()
        m2 = m ** 2

        obvs = torch.stack([e, e2, m, m2], axis=0)
        #####################

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * b)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        state = torch.cat([(s * flip), b], axis=1)
        return state, obvs

class isingCA(nn.Module):
    def __init__(self, RES, CHANNELS=1, BETA=1, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.radius = RADIUS

        self.rule = Rule(RES, BETA, CHANNELS, RADIUS)

    def initGrid(self):
        shape = self.rule.res
        rand = (torch.rand(1, self.channels + 1, shape[0], shape[1]) > torch.rand(1)) * 2. - 1.
        rand[:, -1, ...] = torch.ones_like(rand[:, -1, ...]) * self.rule.beta

        obvs = torch.zeros(1, self.rule.num_obvs, shape[0], shape[1])
        return rand.cuda(), obvs.cuda()

    def forward(self, state):
        return self.rule(state)
