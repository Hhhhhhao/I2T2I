import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


class RLLoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(RLLoss, self).__init__()
        self.eps = eps

    def forward(self, rewards, props):
        loss = rewards * torch.log(torch.clamp(props, min=self.eps, max=1.0))
        # loss = rewards * props
        loss = -torch.mean(loss)
        return loss


class EvaluatorLoss(torch.nn.Module):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, evaluator_outputs, generator_outputs, other_outputs):
        batch_size = evaluator_outputs.size(0)
        true_labels = torch.ones((batch_size, 1)).long()
        fake_labels = torch.zeros((batch_size, 1)).long()

        if torch.cuda.is_available():
            true_labels = true_labels.cuda()
            fake_labels = fake_labels.cuda()

        true_loss = self.loss(evaluator_outputs, true_labels)
        fake_loss = self.loss(generator_outputs, fake_labels)
        other_loss = self.loss(other_outputs, fake_labels)
        loss = true_loss + self.alpha * fake_loss + self.beta * other_loss
        return loss
