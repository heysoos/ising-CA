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

        ## temperature stuff
        self.temp_adapt = False
        self.alpha = 0.05  # update rate
        self.h = 1e-1  # magnetization coef (growth coef)
        self.eps = 2.00e-2  # decay coef
        self.D = 0.95 #2. * self.eps  # diffusion coef

        self.m_pow = 2.
        self.temp_pow = 1.
        self.temp_kernel_size = 1

        ## connectivity stuff
        self.h_J = 0.001
        self.eps_J = 1
        self.adapt_lr = 1.

        self.num_obvs = 4
        self.obv_decay = 0.95
        self.obv_radius = 1 # for observations that are macroscopic in nature

        ###### LOCAL CONNECTIVITY ######
        self.numel = RES[0] * RES[1]

        # radial kernel
        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()
        exp_rm = torch.exp(-rm ** 2)
        condition = rm < 1.
        null = torch.zeros_like(rm).cuda()
        self.rm = torch.where(condition, exp_rm, null).unsqueeze(0).unsqueeze(-1)

        # nearest_neighbours = torch.zeros(1, Rk, Rk, self.numel).cuda()

        # nearest_neighbours[:, RADIUS, :, :] = 1.
        # nearest_neighbours[:, :, RADIUS, :] = 1.
        # nearest_neighbours[:, RADIUS, RADIUS, :] = 0

        nearest_neighbours = torch.rand(1, Rk, Rk, self.numel).cuda()
        nearest_neighbours[:, RADIUS, RADIUS, :] = 0

        # self.nearest_neighbours = (nearest_neighbours * self.rm).reshape(1, -1, self.numel)
        self.nearest_neighbours = (nearest_neighbours).reshape(1, -1, self.numel)
        ####################################


        ###### ORIGINAL CONNECTIVITY ######
        # nearest_neighbours = torch.zeros(1, CHANNELS, Rk, Rk).cuda()
        #
        # nearest_neighbours[:, :, RADIUS, :] = 1.
        # nearest_neighbours[:, :, :, RADIUS] = 1.
        # nearest_neighbours[:, :, RADIUS, RADIUS] = 0
        #
        # self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)
        ####################################

    def forward(self, x, obvs):

        s = x[:, :self.channels, ...]
        b = x[:, [-1], ...]

        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')
        # b_pad = F.pad(b, (Rk, Rk, Rk, Rk), mode='circular')

        sJs = 2 * F.unfold(s, 1) * (F.unfold(s_pad, 2 * Rk + 1) * self.nearest_neighbours)
        # Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        delta_e = sJs.sum(dim=1).view(*shape)
        E = (-0.5 * delta_e)

        ### OBSERVATIONS ###
        e = obvs[:, [0], ...]
        e2 = obvs[:, [1], ...]
        c = obvs[:, [2], ...]
        m = obvs[:, [3], ...]

        e = self.obv_decay * e + (1 - self.obv_decay) * E
        e2 = self.obv_decay * e2 + (1 - self.obv_decay) * E**2
        c = self.obv_decay * c + (1 - self.obv_decay) * (e2 - e**2) * b**2

        m = self.obv_decay * m + (1 - self.obv_decay) * s

        # m = F.avg_pool2d(s, )
        obvs = torch.cat([e, e2, c, m], axis=1)
        #####################

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * b)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        if self.temp_adapt and torch.rand(1) > 0.9:


            ### TEMPERATURE STUFF ###
            temp_rad = self.temp_kernel_size*Rk
            temp_kernel_size = 2*temp_rad + 1
            pads = tuple([Rk * self.temp_kernel_size for i in range(4)])

            s_unfold = F.unfold(s_pad, 2*Rk + 1) # localize measurements
            sm = (s_unfold.mean(dim=1)).reshape(shape)

            b_tpad = F.pad(b, pads, mode='circular')
            T_pad = 1. / b_tpad
            T = T_pad[..., temp_rad:-temp_rad, temp_rad:-temp_rad]
            diff_T = (F.avg_pool2d(T_pad, temp_kernel_size, stride=1) - T)

            # diffuse along connectivity
            # diff_T = ( (F.unfold(T_pad, 2*Rk+1) - F.unfold(T, 1))
            #            * self.nearest_neighbours).mean(dim=1).view(*shape)

            # newT = self.h * sm ** 2 - self.eps * T + self.D * diff_T
            deltaT = self.h * sm.abs() ** self.m_pow \
                     - self.eps * T ** self.temp_pow +\
                     self.D * diff_T
            newT = T + deltaT
            newT = (1 - self.alpha) * T + self.alpha * newT

            b = 1 / newT
            ##########################

            ##### CONNECTIVITY ADAPTIVITY #####
            # s_i = F.unfold(F.pad(m, [Rk for i in range(4)], mode='circular'), 2*Rk + 1)
            # m_local = s_i.mean(dim=1)

            # dJ = (self.h_J - self.eps_J * m_local.abs()) * self.rm.reshape(1, -1, 1)
            growth = self.eps_J * torch.softmax(sJs * (sJs > 0.), dim=1)
            decay = self.h_J * F.unfold(delta_e, 1)
            dJ = (growth - decay) * torch.sign(self.nearest_neighbours)
            dJ *= torch.rand_like(dJ) > 0.5
            # dJ = torch.softmax(dJ, dim=1)
            # dJ = dJ - dJ.mean(dim=1)

            new_J = self.nearest_neighbours + \
                    dJ * self.adapt_lr  # * torch.rand_like(self.nearest_neighbours) # stochastic dJ
            new_J[:, Rk * (2 * Rk + 1) + Rk, :] = 0.  # set center pixel to 0.
            ###################################

            self.nearest_neighbours = (
                    (1 - self.alpha) * self.nearest_neighbours + self.alpha * new_J
            ).clip(min=-1., max=1.)

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

    def forward(self, state, obvs):
        return self.rule(state, obvs)
