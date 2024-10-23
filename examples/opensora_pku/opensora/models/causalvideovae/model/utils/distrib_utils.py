from mindspore import mint


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.mean, self.logvar = mint.split(parameters, [parameters.shape[1] // 2, parameters.shape[1] // 2], dim=1)
        self.logvar = mint.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = mint.exp(0.5 * self.logvar)
        self.var = mint.exp(self.logvar)
        self.stdnormal = mint.normal
        if self.deterministic:
            self.var = self.std = mint.zeros_like(self.mean, dtype=self.mean.dtype)

    def sample(self):
        x = self.mean + self.std * self.stdnormal(size=self.mean.shape)
        return x

    def mode(self):
        return self.mean
