import torch as ch


class AttackerStep:
    def __init__(self, orig_input, eps, step_size, use_grad=True, min_value=0, max_value=1):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.min_value = min_value
        self.max_value = max_value


class L2Step(AttackerStep):
    def project(self, x):
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, self.min_value, self.max_value)

    def step(self, x, g):
        # Scale g so that each element of the batch is at least norm 1
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(new_x, self.min_value, self.max_value)
