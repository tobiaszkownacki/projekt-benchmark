import torch
from torch.optim import Optimizer

# Gravity Optimizer
# Source: "Gravity Optimizer: a Kinematic Approach on Optimization in Deep Learning"
# arxiv: https://arxiv.org/abs/2101.09192
# GitHub repo: https://github.com/dariush-bahrami/gravity.optimizer

class Gravity(Optimizer):
    def __init__(self, params, lr=0.1, alpha=0.01, beta=0.9):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta}")
        defaults = dict(lr=lr, alpha=alpha, beta=beta, t=0)
        super(Gravity, self).__init__(params, defaults)
        self.__init_v()

    def __init_v(self):
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            std = alpha / lr
            for p in group["params"]:
                state = self.state[p]
                state["v"] = torch.empty_like(p).normal_(mean=0.0, std=std)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            beta = group["beta"]
            group["t"] += 1
            t = group["t"]
            beta_hat = (beta * t + 1) / (t + 2)
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("Sparse gradients are not supported")
                    state = self.state[p]
                    m = 1 / torch.abs(p.grad).max()
                    zeta = p.grad / (1 + (p.grad / m) ** 2)
                    state["v"] = beta_hat * state["v"] + (1 - beta_hat) * zeta
                    p -= lr * state["v"]
        return loss