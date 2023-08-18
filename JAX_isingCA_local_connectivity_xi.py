import jax
import jax.numpy as jnp
from jax import random, lax, jit
from functools import partial


class Rule:
    def __init__(self, key, RES, BETA=1, RADIUS=1):
        self.key = key

        self.beta = BETA
        self.radius = RADIUS
        self.res = RES
        Rk = 2 * RADIUS + 1

        self.J_adapt = True
        self.adapt_lr = 0.5
        self.alpha = 0.9
        self.h = 0.01
        self.eps = 0.01
        self.max_weight = 4.

        self.trace_memory = 0.995

        self.num_obvs = 4
        self.obv_decay = 0.95

        self.numel = RES[0] * RES[1]

        # make random connections
        nearest_neighbours = random.uniform(key, (Rk, Rk, RES[0], RES[1]))
        nearest_neighbours = nearest_neighbours.at[RADIUS, RADIUS, :, :].set(0)
        self.nearest_neighbours = nearest_neighbours.reshape(1, -1, RES[0], RES[1])

    @partial(jit, static_argnums=(0, 2, 3))
    def forward(self, x, key, make_obv=False):
        s = x[:, [0], ...]
        tr = x[:, [1], ...]
        b = x[:, [-1], ...]
        shape = (1, 1, self.res[0], self.res[1])
        Rk = self.radius
        key, *subkeys = random.split(key, 4)

        if make_obv:
            flip, obvs = self.ising_update(s, b, make_obv, subkeys[0])
        else:
            flip = self.ising_update(s, b, make_obv, subkeys[0])

        # rng for adaptation
        rng_adapt = random.uniform(subkeys[1], (1, ))
        adapt_thresh = 0.0
        adapt_cond = bool(self.J_adapt and (rng_adapt > adapt_thresh))

        # if adapt_cond:
        #     self.J_adaptation(tr)

        self.nearest_neighbours = jax.lax.cond(adapt_cond,
                                               self.J_adaptation,
                                               lambda x, y: self.nearest_neighbours,
                                               tr, subkeys[2])

        tr = self.trace_memory * tr + (1 - self.trace_memory) * s

        if make_obv:
            return jnp.concatenate([(s * flip), tr, b], axis=1), obvs, key
        else:
            return jnp.concatenate([(s * flip), tr, b], axis=1), key

    @partial(jit, static_argnums=(0, 2))
    def J_adaptation(self, tr, key):
        Rk = self.radius
        shape = (1, 1, self.res[0], self.res[1])

        s_i = jax_unfold(tr, 1)
        tr_pad = jnp.pad(tr, [(0, 0), (0, 0), (Rk, Rk), (Rk, Rk)], mode='wrap')
        s_j = jax_unfold(tr_pad, 2 * Rk + 1)

        growth = self.h / Rk * (1 - jnp.abs(s_i)) * (1 - jnp.abs(s_j))  # correlate
        decay = self.eps * (s_j * s_i)  # decorrelate if mag.
        dJ = (growth - decay)  # * self.rm.reshape(1, -1, 1)

        key, subkey = random.split(key)
        conn_shape = (1, int((2*Rk+1)**2), shape[-2], shape[-1])
        adapt_mask = random.uniform(subkey, conn_shape) > 0.5

        new_J = self.nearest_neighbours + \
                dJ * self.adapt_lr * adapt_mask  # stochastic dJ

        # set center pixel to 0.
        new_J = new_J.at[:, :, Rk, Rk].set(0.)
        # set corners to 0
        new_J = new_J.at[:, :, -1, -1].set(0.)
        new_J = new_J.at[:, :, -1, 0].set(0.)
        new_J = new_J.at[:, :, 0, -1].set(0.)
        new_J = new_J.at[:, :, 0, 0].set(0.)

        nearest_neighbours = ((1 - self.alpha) * self.nearest_neighbours + self.alpha * new_J)

        return nearest_neighbours

    @partial(jit, static_argnums=(0, 3, 4))
    def ising_update(self, s, b, make_obv, key):
        shape = (1, 1, self.res[0], self.res[1])
        Rk = self.radius

        s_pad = jnp.pad(s, [(0, 0), (0, 0), (Rk, Rk), (Rk, Rk)], mode='wrap')

        sJs = 2 * jax_unfold(s, 1) * (jax_unfold(s_pad, 2 * Rk + 1) * self.nearest_neighbours)
        delta_e = sJs.sum(axis=1).reshape(*shape)
        E = (-0.5 * delta_e)

        if make_obv:
            ### OBSERVATIONS ###
            e = E.mean() / 4.
            e2 = e ** 2

            m = s.mean()
            m2 = m ** 2

            obvs = jnp.stack([e, e2, m, m2], axis=0)
        #####################

        definite_flip = delta_e <= 0
        p = jnp.exp(-delta_e * b)
        p = jnp.where(definite_flip, jnp.ones_like(s), p)

        key, *subkeys = random.split(key, 4)
        flip_rand = random.uniform(subkeys[0], shape)
        dropout_mask = (random.uniform(subkeys[1], shape) > 0.5)

        flip = -2. * jnp.logical_and(flip_rand < p, dropout_mask) + 1

        if make_obv:
            return flip, obvs
        else:
            return flip



class isingCA:
    def __init__(self, key, RES, BETA=1, RADIUS=1):
        self.radius = RADIUS
        self.rule = Rule(key, RES, BETA, RADIUS)

    def initGrid(self, key):
        shape = self.rule.res
        rand_s = random.bernoulli(key, p=0.5, shape=(1, 1, shape[0], shape[1])) * 2. - 1.
        rand_tr = jnp.zeros((1, 1, shape[0], shape[1]), dtype=jnp.float32)
        rand_b = jnp.ones((1, 1, shape[0], shape[1]), dtype=jnp.float32) * self.rule.beta

        return jnp.concatenate([rand_s, rand_tr, rand_b], axis=1)

    def forward(self, x, key, make_obv=False):
        return self.rule.forward(x, key, make_obv)

def jax_unfold(arr, k_size):
    arr_unf = jax.lax.conv_general_dilated_patches(
        lhs=arr,
        filter_shape=(k_size,k_size),
        window_strides=(1,1),
        padding = 'VALID',
        dimension_numbers  = ('NCWH', 'WHIO', 'NCWH'))
    return arr_unf
