import mindspore.ops as ops


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.mean, self.logvar = ops.Split(axis=1, output_num=2)(parameters)
        self.logvar = ops.clip_by_value(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = ops.exp(0.5 * self.logvar)
        self.var = ops.exp(self.logvar)
        self.stdnormal = ops.StandardNormal()
        if self.deterministic:
            self.var = self.std = ops.zeros_like(self.mean, dtype=self.mean.dtype)

    def sample(self):
        x = self.mean + self.std * self.stdnormal(self.mean.shape)
        return x

    def mode(self):
        return self.mean
