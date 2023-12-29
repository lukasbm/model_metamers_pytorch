import torch as ch

from replicate.attack_steps import L2Step


class Attacker(ch.nn.Module):
    def __init__(self, model: ch.nn.Module, dataset, preproc):
        super(Attacker, self).__init__()
        self.preproc = preproc
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
    def __init__(self, model, dataset, preproc):
        super(AttackerModel, self).__init__()
        self.preproc = preproc
        self.model = model
        self.attacker = Attacker(model, dataset, preproc)

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
