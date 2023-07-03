import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, BETA=1, RADIUS=2):
        super().__init__()
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        nearest_neighbours = torch.zeros(1, 1, Rk, Rk).cuda()
        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0

        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)

    def forward(self, x):

        Rk = self.radius
        s = F.pad(x[:, 0, ...], (Rk, Rk, Rk, Rk), mode='circular')
        momentum = F.pad(x[:, 1:, ...], (Rk, Rk, Rk, Rk), mode='circular')
        Js = F.conv2d(s, self.nearest_neighbours, padding=0)
        Hk = 4 * (momentum[0, 0] + 2 * momentum[0, 1])  # the 4 is to match this with the number of neighbours
        delta_e = 2 * x * Js

        (12 - Hk) - delta_e

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * self.beta)
        p = torch.where(definite_flip, torch.ones_like(x), p)

        rand = torch.rand_like(x)

        dropout_mask = (torch.rand_like(x[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        return (x * flip)

class isingCA(nn.Module):
    def __init__(self, BETA=1, RADIUS=1):
        super().__init__()
        self.radius = RADIUS

        self.rule = Rule(BETA, RADIUS)

    def initGrid(self, shape):
        rand = (np.random.rand(1, 3, shape[0], shape[1]) > 0.5) * 2. - 1.
        return torch.cuda.FloatTensor(rand)

    def forward(self, x):
        return self.rule(x)

    def cleanup(self):
        del self.psi
