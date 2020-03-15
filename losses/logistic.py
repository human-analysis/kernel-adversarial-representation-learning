
from torch import nn

__all__ = ['Logistic']


# REVIEW: does this have to inherit nn.Module?
class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss
