import torch
from torch import Tensor

from model_evaluator import ModelEvaluator

class AdamOptimizer:
    def __init__(
        self,
        initial_params: Tensor,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        **config,
    ):
        self.config = config
        self.params = initial_params.detach().clone()

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = torch.zeros_like(initial_params)
        self.v = torch.zeros_like(initial_params)
        self.t = 0

    def step(self, evaluator: ModelEvaluator) -> bool:
        loss, grad = evaluator.evaluate_with_grad(self.params)
        self.t += 1

        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.params

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.params = self.params - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        return False