"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module contains the abstract class AttackerStep as well as a few subclasses. 

AttackerStep is a generic way to implement optimizers specifically for use with
:class:`robustness.attacker.AttackerModel`. In general, except for when you want
to :ref:`create a custom optimization method <adding-custom-steps>`, you probably do not need to
import or edit this module and can just think of it as internal.
"""

import torch as ch

class AttackerStep:
    """
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    """

    def __init__(self, orig_input, eps, step_size, use_grad=True,
                 min_value=0, max_value=1):
        """
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
            min_value (float): the minimum value in the dataset (to clamp)
            max_value (float): the maximum value in the dataset (to clamp)
        """
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.min_value = min_value
        self.max_value = max_value

    def to_image(self, x):
        return x


class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """

    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, self.min_value, self.max_value)

    def step(self, x, g):
        """
        """
        # Scale g so that each element of the batch is at least norm 1
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(new_x, self.min_value, self.max_value)
