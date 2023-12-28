"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module houses the :class:`robustness.attacker.Attacker` and
:class:`robustness.attacker.AttackerModel` classes. 

:class:`~robustness.attacker.Attacker` is an internal class that should not be
imported/called from outside the library.
:class:`~robustness.attacker.AttackerModel` is a "wrapper" class which is fed a
model and adds to it adversarial attack functionalities as well as other useful
options. See :meth:`robustness.attacker.AttackerModel.forward` for documentation
on which arguments AttackerModel supports, and see
:meth:`robustness.attacker.Attacker.forward` for the arguments pertaining to
adversarial examples specifically.

For a demonstration of this module in action, see the walkthrough
":doc:`../example_usage/input_space_manipulation`"

**Note 1**: :samp:`.forward()` should never be called directly but instead the
AttackerModel object itself should be called, just like with any
:samp:`nn.Module` subclass.

**Note 2**: Even though the adversarial example arguments are documented in
:meth:`robustness.attacker.Attacker.forward`, this function should never be
called directly---instead, these arguments are passed along from
:meth:`robustness.attacker.AttackerModel.forward`.
"""

import torch as ch

from robustness.tools import helpers
from .attack_steps import L2Step


class Attacker(ch.nn.Module):
    def __init__(self, model: ch.nn.Module, dataset):
        super(Attacker, self).__init__()
        self.preproc = helpers.GraphPreprocessing(dataset)
        self.model = model
        self.dataset_min_value = dataset.min_value
        self.dataset_max_value = dataset.max_value

    def forward(
            self,
            x,
            target,
            custom_loss,
            *_,
            eps,
            step_size,
            iterations,
            targeted=False,
            should_preproc=True,
            orig_input=None,
    ):
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None:
            orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        step = L2Step(eps=eps, orig_input=orig_input, step_size=step_size, min_value=self.dataset_min_value,
                      max_value=self.dataset_max_value)

        def calc_loss(inp, targ):
            """
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            """
            if should_preproc:
                inp = self.preproc(inp)
            return custom_loss(self.model, inp, targ)

        # Main function for making adversarial examples using PGD
        def get_adv_examples(x):
            # PGD iterates (we will optimize x)
            for _ in range(iterations):
                x = x.clone().detach().requires_grad_(True)

                # calculating the loss also does the forward pass.
                losses, _ = calc_loss(x, target)
                assert losses.shape[0] == x.shape[0], 'Shape of losses must match input!'

                loss = ch.mean(losses)

                # calculate gradient
                if step.use_grad:
                    grad, = ch.autograd.grad(m * loss, [x])
                else:
                    grad = None

                with ch.no_grad():
                    # update the metamer
                    x = step.step(x, grad)
                    x = step.project(x)

            return x.clone().detach()

        return get_adv_examples(x)


class AttackerModel(ch.nn.Module):
    def __init__(self, model, dataset):
        super(AttackerModel, self).__init__()
        self.preproc = helpers.GraphPreprocessing(dataset)
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, make_adv=False, with_image=True, **attacker_kwargs):
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            self.train(prev_training)  # restore mode
            inp = adv

        if with_image:
            preproc_inp = self.preproc(inp)
            output = self.model(preproc_inp)
        else:
            output = None

        return output, inp
