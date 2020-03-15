# classification.py

__all__ = ['Classification', 'Top1Classification', 'LogClassification', 'BinClassification']


import torch
import torch.nn as nn
Sig = nn.Sigmoid()


class Classification:
    def __init__(self, topk=(1,)):
        self.topk = topk

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        maxk = max(self.topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Top1Classification:
    def __init__(self):
        pass

    def __call__(self, output, target):
        batch_size = target.size(0)

        pred = output.data.max(1)[1].view(-1, 1)
        res = pred.eq(target.data).cpu().sum().float() * 100 / batch_size

        return res


class LogClassification:
    def __init__(self):
        pass

    def __call__(self, output, target):
        batch_size = target.size(0)

        output = Sig(output)
        pred = torch.ge(output, 0.5).float()

        res = pred.eq(target.data).cpu().sum().float() * 100 / batch_size

        # import pdb
        # pdb.set_trace()

        return res


class BinClassification:
    def __init__(self):
        pass

    def __call__(self, output, target):
        batch_size = target.size(0)

        pred = torch.ge(output, 0.5).float()

        res = pred.eq(target.data).cpu().sum().float() * 100 / batch_size

        # import pdb
        # pdb.set_trace()

        return res