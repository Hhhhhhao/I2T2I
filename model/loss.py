import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


class RLLoss(torch.nn.Module):

    def forward(self, rewards, props):
        loss = rewards * torch.log(props)
        loss = -torch.sum(loss)
        return loss


class EvaluatorLoss(torch.nn.Module):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, evaluator_outputs, generator_outputs, other_outputs):
        generator_outputs = 1 - generator_outputs
        other_outputs = 1 - other_outputs
        temp = 1e-5
        evaluator_outputs = evaluator_outputs.clamp(min=temp)
        generator_outputs = generator_outputs.clamp(min=temp)
        other_outputs = other_outputs.clamp(min=temp)
        t1, t2, t3 = torch.log(evaluator_outputs), self.alpha * torch.log(generator_outputs), self.beta * torch.log(
            other_outputs)
        result = t1 + t2 + t3
        result = -result
        return result.mean()
