from __future__ import division
import numpy as np


class RandomProcess(object):
    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed):
        self.r = np.random.RandomState(seed)

    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, **kwargs):
        super(AnnealedGaussianProcess, self).__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0
        
        self.c = sigma
        
        if sigma_min is not None:
            assert np.array(sigma > sigma_min).all()
            self.m = -(sigma - sigma_min) / float(n_steps_annealing)
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        return np.maximum(self.sigma_min, self.m * float(self.n_steps) + self.c)


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, size=1, **kwargs):
        super(GaussianWhiteNoiseProcess, self).__init__(**kwargs)
        self.size = size

    def sample(self):
        self.n_steps += 1
        return self.r.normal(self.mu, self.current_sigma, self.size)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, dt=1e-2, x0=None, size=1, **kwargs):
        super(OrnsteinUhlenbeckProcess, self).__init__(**kwargs)
        assert dt > 0

        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * self.r.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
